use chumsky::{prelude::*, stream::Stream};
use std::collections::HashMap;
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

    let ctrl = one_of("()[]{};,").map(Token::ControlChar);

    let identifier = text::ident().map(|ident: String| match ident.as_str() {
        "func" => Token::Func,
        "var" => Token::Var,
        "if" => Token::If,
        "else" => Token::Else,
        "true" => Token::Bool(true),
        "false" => Token::Bool(false),
        "null" => Token::Null,
        _ => Token::Identifier(ident),
    });

    let token = num
        .or(str_)
        .or(op)
        .or(ctrl)
        .or(identifier)
        .recover_with(skip_then_retry_until([]));
    let comment = just("//").then(take_until(just('\n'))).padded();
    token
        .padded_by(comment.repeated())
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
}

// A function node in the AST.
#[derive(Debug)]
pub struct Func {
    pub args: Vec<Spanned<String>>,
    pub body: Spanned<Expr>,
    pub name: Spanned<String>,
    pub span: Span,
}

/*fn expr_parser() -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone {
    recursive(|expr| {

    })
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
}*/

pub fn parse(src: &str) -> (Option<HashMap<String, Func>>, Vec<Simple<String>>, Vec<ImCompleteSemanticToken>) {
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
        /*let (ast, parse_errs) =
            funcs_parser().parse_recovery(Stream::from_iter(len..len + 1, tokens.into_iter()));*/
        let ast: Option<HashMap<String, Func>> = None;
        let parse_errs : Vec<Simple<Token>> = vec![];
        (ast, parse_errs, semantic_tokens)
    } else {
        (None, Vec::new(), vec![])
    };
    (None, vec![], vec![])
}