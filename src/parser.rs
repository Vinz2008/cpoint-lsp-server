use chumsky::error::Cheap;
use chumsky::text::{newline, ident};
use chumsky::{prelude::*, stream::Stream};
use std::collections::HashMap;
use std::fmt;
use tower_lsp::lsp_types::SemanticTokenType;

use crate::semantic_token::LEGEND_TYPE;

pub type Span = std::ops::Range<usize>;

pub type Spanned<T> = (T, Span);

#[derive(Debug)]
pub struct ImCompleteSemanticToken {
    pub start: usize,
    pub length: usize,
    pub token_type: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Token {
    Null,
    Bool(bool),
    Str(String),
    Num(String),
    Operator(String),
    Identifier(String),
    ControlChar(char),
    Func,
    Var,
    If,
    Else,
    For,
    While,
//    Import,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Null => write!(f, "null"),
            Token::Bool(x) => write!(f, "{}", x),
            Token::Num(n) => write!(f, "{}", n),
            Token::Str(s) => write!(f, "{}", s),
            Token::Operator(s) => write!(f, "{}", s),
            Token::ControlChar(c) => write!(f, "{}", c),
            Token::Identifier(s) => write!(f, "{}", s),
            Token::Func => write!(f, "func"),
            Token::Var => write!(f, "var"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::For => write!(f, "for"),
            Token::While => write!(f, "while"),
            //Token::Import => write!(f, "import"),
        }
    }
}

fn lexer() -> impl Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
    let num = text::int(10)
        .chain::<char, _, _>(just('.').chain(text::digits(10)).or_not().flatten())
        .collect::<String>()
        .map(Token::Num);

    let str_ = just('"')
        .ignore_then(filter(|c| *c != '"').repeated())
        .then_ignore(just('"'))
        .collect::<String>()
        .map(Token::Str);

    let op = one_of("+-*/!=<>|&")
        .repeated()
        .at_least(1)
        .collect::<String>()
        .map(Token::Operator);

    let ctrl = one_of("()[]{};,./").map(Token::ControlChar);

    let identifier = text::ident().map(|ident: String| match ident.as_str() {
        "func" => Token::Func,
        "var" => Token::Var,
        "if" => Token::If,
        "else" => Token::Else,
        "true" => Token::Bool(true),
        "false" => Token::Bool(false),
        "null" => Token::Null,
        "for" => Token::For,
        "while" => Token::While,
//        "import" => Token::Import,
        _ => Token::Identifier(ident),
    });

    let token = num
        .or(str_)
        .or(op)
        .or(ctrl)
        .or(identifier)
        .recover_with(skip_then_retry_until([]));
    let comment = just("//").then(take_until(just('\n'))).padded();
    //let import = just("import").then(take_until(just('\n')));
    token
        .padded_by(comment.repeated())
        //.padded_by(import.repeated())
        .map_with_span(|tok, span| (tok, span))
        .padded()
        .repeated()
}

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<Value>),
    Func(String),
}

#[derive(Clone, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    NotEq,
}

// An expression node in the AST. Children are spanned so we can generate useful runtime errors.
#[derive(Debug)]
pub enum Expr {
    Error,
    Value(Value),
    List(Vec<Spanned<Self>>),
    Local(Spanned<String>),
    Var(String, Box<Spanned<Self>>, Box<Spanned<Self>>, Span),
    Then(Box<Spanned<Self>>, Box<Spanned<Self>>),
    Binary(Box<Spanned<Self>>, BinaryOp, Box<Spanned<Self>>),
    Call(Box<Spanned<Self>>, Spanned<Vec<Spanned<Self>>>),
    If(Box<Spanned<Self>>, Box<Spanned<Self>>, Box<Spanned<Self>>),
//    For(Box<Spanned<Self>>, Box<Spanned<Self>>)
}

#[derive(Debug)]
pub enum TopLevelExpr {
    Function(Func),
    Import(Import),
}

// A function node in the AST.
#[derive(Debug)]
pub struct Func {
    pub args: Vec<Spanned<String>>,
    pub body: Spanned<Expr>,
    pub name: Spanned<String>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Import {
    pub path: Spanned<String>,
    pub span: Span,
}

fn expr_parser() -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone {
    recursive(|expr| {
        let raw_expr = recursive(|raw_expr| {
            let val = filter_map(|span, tok| match tok {
                Token::Null => Ok(Expr::Value(Value::Null)),
                Token::Bool(x) => Ok(Expr::Value(Value::Bool(x))),
                Token::Num(n) => Ok(Expr::Value(Value::Num(n.parse().unwrap()))),
                Token::Str(s) => Ok(Expr::Value(Value::Str(s))),
                _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
            })
            .labelled("value");

            let ident = filter_map(|span, tok| match tok {
                Token::Identifier(ident) => Ok((ident, span)),
                _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
            })
            .labelled("identifier");

            // A list of expressions
            let items = expr
                .clone()
                .chain(
                    just(Token::ControlChar(','))
                        .ignore_then(expr.clone())
                        .repeated(),
                )
                .then_ignore(just(Token::ControlChar(',')).or_not())
                .or_not()
                .map(|item| item.unwrap_or_default());

            // A let expression
            let var_ = just(Token::Var)
                .ignore_then(ident)
                .then_ignore(just(Token::Operator("=".to_string())))
                .then(raw_expr)
                .then(expr.clone())
                .map(|((name, val), body)| {
                    Expr::Var(name.0, Box::new(val), Box::new(body), name.1)
                });

            let list = items
                .clone()
                .delimited_by(just(Token::ControlChar('[')), just(Token::ControlChar(']')))
                .map(Expr::List);

            // 'Atoms' are expressions that contain no ambiguity
            let atom = val
                .or(ident.map(Expr::Local))
                .or(var_)
                .or(list)
                .map_with_span(|expr, span| (expr, span))
                // Atoms can also just be normal expressions, but surrounded with parentheses
                .or(expr
                    .clone()
                    .delimited_by(just(Token::ControlChar('(')), just(Token::ControlChar(')'))))
                // Attempt to recover anything that looks like a parenthesised expression but contains errors
                .recover_with(nested_delimiters(
                    Token::ControlChar('('),
                    Token::ControlChar(')'),
                    [
                        (Token::ControlChar('['), Token::ControlChar(']')),
                        (Token::ControlChar('{'), Token::ControlChar('}')),
                    ],
                    |span| (Expr::Error, span),
                ))
                // Attempt to recover anything that looks like a list but contains errors
                .recover_with(nested_delimiters(
                    Token::ControlChar('['),
                    Token::ControlChar(']'),
                    [
                        (Token::ControlChar('('), Token::ControlChar(')')),
                        (Token::ControlChar('{'), Token::ControlChar('}')),
                    ],
                    |span| (Expr::Error, span),
                ));

            // Function calls have very high precedence so we prioritise them
            let call = atom
                .then(
                    items
                        .delimited_by(just(Token::ControlChar('(')), just(Token::ControlChar(')')))
                        .map_with_span(|args, span| (args, span))
                        .repeated(),
                )
                .foldl(|f, args| {
                    let span = f.1.start..args.1.end;
                    (Expr::Call(Box::new(f), args), span)
                });

            // Product ops (multiply and divide) have equal precedence
            let op = just(Token::Operator("*".to_string()))
                .to(BinaryOp::Mul)
                .or(just(Token::Operator("/".to_string())).to(BinaryOp::Div));
            let product = call
                .clone()
                .then(op.then(call).repeated())
                .foldl(|a, (op, b)| {
                    let span = a.1.start..b.1.end;
                    (Expr::Binary(Box::new(a), op, Box::new(b)), span)
                });

            // Sum ops (add and subtract) have equal precedence
            let op = just(Token::Operator("+".to_string()))
                .to(BinaryOp::Add)
                .or(just(Token::Operator("-".to_string())).to(BinaryOp::Sub));
            let sum = product
                .clone()
                .then(op.then(product).repeated())
                .foldl(|a, (op, b)| {
                    let span = a.1.start..b.1.end;
                    (Expr::Binary(Box::new(a), op, Box::new(b)), span)
                });

            // Comparison ops (equal, not-equal) have equal precedence
            let op: chumsky::combinator::Or<chumsky::combinator::To<chumsky::primitive::Just<Token, Token, _>, Token, BinaryOp>, chumsky::combinator::To<chumsky::primitive::Just<Token, Token, _>, Token, BinaryOp>> = just(Token::Operator("==".to_string()))
                .to(BinaryOp::Eq)
                .or(just(Token::Operator("!=".to_string())).to(BinaryOp::NotEq));

            sum.clone()
                .then(op.then(sum).repeated())
                .foldl(|a, (op, b)| {
                    let span = a.1.start..b.1.end;
                    (Expr::Binary(Box::new(a), op, Box::new(b)), span)
                })
        });
        /*let ident = filter_map(|span, tok| match tok {
            Token::Identifier(ident) => Ok((ident, span)),
            _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
        })
        .labelled("identifier");*/

        let items = expr
            .clone()
            .chain(
                just(Token::ControlChar(','))
                    .ignore_then(expr.clone())
                    .repeated(),
            )
            .then_ignore(just(Token::ControlChar(',')).or_not())
            .or_not()
            .map(|item| item.unwrap_or_default());
        // Blocks are expressions but delimited with braces
        let block = expr
            .clone()
            .delimited_by(just(Token::ControlChar('{')), just(Token::ControlChar('}')))
            // Attempt to recover anything that looks like a block but contains errors
            .recover_with(nested_delimiters(
                Token::ControlChar('{'),
                Token::ControlChar('}'),
                [
                    (Token::ControlChar('('), Token::ControlChar(')')),
                    (Token::ControlChar('['), Token::ControlChar(']')),
                ],
                |span| (Expr::Error, span),
            ));

        let if_ = recursive(|if_| {
            just(Token::If)
                .ignore_then(expr.clone())
                .then(block.clone())
                .then(
                    just(Token::Else)
                        .ignore_then(block.clone().or(if_))
                        .or_not(),
                )
                .map_with_span(|((cond, a), b), span| {
                    (
                        Expr::If(
                            Box::new(cond),
                            Box::new(a),
                            Box::new(match b {
                                Some(b) => b,
                                // If an `if` expression has no trailing `else` block, we magic up one that just produces null
                                None => (Expr::Value(Value::Null), span.clone()),
                            }),
                        ),
                        span,
                    )
                })
        });

        /*let for_ = recursive(|for_| {
            just(Token::For)
            .then(items)
            .map_with_span(|expr, span| (expr, span))
            .separated_by(just(Token::ControlChar(',')))
            .allow_trailing()
            .then(block.clone())
            .map_with_span(|(items, a), span|{
                (Expr::For(Box::New(items), a), span)
            })
        });*/

        // Both blocks and `if` are 'block expressions' and can appear in the place of statements
        let block_expr = block.or(if_)/*.or(for_)*/.labelled("block");

        let block_chain = block_expr
            .clone()
            .then(block_expr.clone().repeated())
            .foldl(|a, b| {
                let span = a.1.start..b.1.end;
                (Expr::Then(Box::new(a), Box::new(b)), span)
            });

        block_chain
            // Expressions, chained by semicolons, are statements
            .or(raw_expr.clone())
            .then(
                just(Token::ControlChar(';'))
                    .ignore_then(expr.or_not())
                    .repeated(),
            )
            .foldl(|a, b| {
                let span = a.1.clone(); // TODO: Not correct
                (
                    Expr::Then(
                        Box::new(a),
                        Box::new(match b {
                            Some(b) => b,
                            None => (Expr::Value(Value::Null), span.clone()),
                        }),
                    ),
                    span,
                )
            })
    })
}

pub fn toplevel_parser(
) -> impl Parser<Token, HashMap<String, TopLevelExpr>, Error = Simple<Token>> + Clone {
    //let tokens = choice::<_, Simple<char>>(()).padded().repeated();
    //tokens
    //let ident = text::ident().padded();
    let ident = filter_map(|span, tok| match tok {
        Token::Identifier(ident) => Ok(ident),
        _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
    });
    /*let path_filter = recursive(|path| {
        choice((just(Token::ControlChar('.')), just(Token::ControlChar('/'))))
        .repeated()
    }).collect();*/
    /*let path_filter = one_of::<_, _, Cheap<char>>("./")
        .or(any())
        .repeated()
        .at_least(1).labelled("path");*/
    /*let import_parser = just(Token::Import)
    .ignore_then(
        /*filter(|c| *c != nextline())*/
        path_filter
        .map_with_span(|path, span| (path, span)).labelled("import path")
    )
    .map_with_span(|path_tok: Vec<Token>, span| {
        let path : (std::string::String, std::ops::Range<usize>) = ("".to_string(), 0..0);
        for t in path_tok {

        }
         (path.clone(), TopLevelExpr::Import(Import { path, span }))

    })
    .labelled("import");*/
    let args = ident
        .map_with_span(|name, span| (name, span))
        .separated_by(just(Token::ControlChar(',')))
        .allow_trailing()
        .delimited_by(just(Token::ControlChar('(')), just(Token::ControlChar(')')))
        .labelled("function args");
    let func_parser = just(Token::Func)
        .ignore_then(
            ident
                .map_with_span(|name, span| (name, span))
                .labelled("function name"),
        )
        .then(args)
        .then(
            expr_parser()
                .delimited_by(just(Token::ControlChar('{')), just(Token::ControlChar('}')))
                // Attempt to recover anything that looks like a function body but contains errors
                .recover_with(nested_delimiters(
                    Token::ControlChar('{'),
                    Token::ControlChar('}'),
                    [
                        (Token::ControlChar('('), Token::ControlChar(')')),
                        (Token::ControlChar('['), Token::ControlChar(']')),
                    ],
                    |span| (Expr::Error, span),
                )),
        )
        .map_with_span(|((name, args), body), span| {
            (
                name.clone(),
                TopLevelExpr::Function(Func {
                    args,
                    body,
                    name,
                    span,
                }),
            )
        })
        .labelled("function");

    func_parser
        .repeated()
        .try_map(|fs, _| {
            let mut funcs = HashMap::new();
            for ((name, name_span), f) in fs {
                if funcs.insert(name.clone(), f).is_some() {
                    return Err(Simple::custom(
                        name_span,
                        format!("Function '{}' already exists", name),
                    ));
                }
            }
            Ok(funcs)
        })
        .then_ignore(end())
    /*let r#import =
    let top_level_exp = r#import.or(r#fn).padded();
    top_level_exp*/
}

pub fn funcs_parser() -> impl Parser<Token, HashMap<String, Func>, Error = Simple<Token>> + Clone {
    let ident = filter_map(|span, tok| match tok {
        Token::Identifier(ident) => Ok(ident),
        _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
    });
    let args = ident
        .map_with_span(|name, span| (name, span))
        .separated_by(just(Token::ControlChar(',')))
        .allow_trailing()
        .delimited_by(just(Token::ControlChar('(')), just(Token::ControlChar(')')))
        .labelled("function args");
    let func = just(Token::Func)
        .ignore_then(
            ident
                .map_with_span(|name, span| (name, span))
                .labelled("function name"),
        )
        .then(args)
        .then(
            expr_parser()
                .delimited_by(just(Token::ControlChar('{')), just(Token::ControlChar('}')))
                // Attempt to recover anything that looks like a function body but contains errors
                .recover_with(nested_delimiters(
                    Token::ControlChar('{'),
                    Token::ControlChar('}'),
                    [
                        (Token::ControlChar('('), Token::ControlChar(')')),
                        (Token::ControlChar('['), Token::ControlChar(']')),
                    ],
                    |span| (Expr::Error, span),
                )),
        )
        .map_with_span(|((name, args), body), span| {
            (
                name.clone(),
                Func {
                    args,
                    body,
                    name,
                    span,
                },
            )
        })
        .labelled("function");

    func.repeated()
        .try_map(|fs, _| {
            let mut funcs = HashMap::new();
            for ((name, name_span), f) in fs {
                if funcs.insert(name.clone(), f).is_some() {
                    return Err(Simple::custom(
                        name_span,
                        format!("Function '{}' already exists", name),
                    ));
                }
            }
            Ok(funcs)
        })
        .then_ignore(end())
}

pub fn parse(
    src: &str,
) -> (
    Option<HashMap<String, /*Func*/ TopLevelExpr>>,
    Vec<Simple<String>>,
    Vec<ImCompleteSemanticToken>,
) {
    let (tokens, errs) = lexer().parse_recovery(src);
    let (ast, tokenize_errors, semantic_tokens) = if let Some(tokens) = tokens {
        let semantic_tokens = tokens
            .iter()
            .filter_map(|(token, span)| match token {
                Token::Null => None,
                Token::Bool(_) => None,

                Token::Num(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::NUMBER)
                        .unwrap(),
                }),
                Token::Str(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::STRING)
                        .unwrap(),
                }),
                Token::Operator(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::OPERATOR)
                        .unwrap(),
                }),
                Token::ControlChar(_) => None,
                Token::Identifier(_) => None,
                Token::Func => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::KEYWORD)
                        .unwrap(),
                }),
                Token::Var => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::KEYWORD)
                        .unwrap(),
                }),
                // TODO : maybe add a semantic token for top level keywords
                /*Token::Import => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::KEYWORD)
                        .unwrap(),
                }),*/
                Token::If => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::KEYWORD)
                        .unwrap(),
                }),
                Token::For => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::KEYWORD)
                        .unwrap(),
                }),
                Token::While => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::KEYWORD)
                        .unwrap(),
                }),
                Token::Else => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::KEYWORD)
                        .unwrap(),
                }),
            })
            .collect::<Vec<_>>();
        let len = src.chars().count();
        let (ast, parse_errs) =
            /*funcs_parser()*/ toplevel_parser().parse_recovery(Stream::from_iter(len..len + 1, tokens.into_iter()));
        (ast, parse_errs, semantic_tokens)
    } else {
        (None, Vec::new(), vec![])
    };
    let parse_errors = errs
        .into_iter()
        .map(|e| e.map(|c| c.to_string()))
        .chain(
            tokenize_errors
                .into_iter()
                .map(|e| e.map(|tok| tok.to_string())),
        )
        .collect::<Vec<_>>();

    (ast, parse_errors, semantic_tokens)
}
