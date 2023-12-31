use crate::parser::{Expr, Func, ImCompleteSemanticToken, Import, Spanned, TopLevelExpr};
use std::collections::HashMap;
use tower_lsp::lsp_types::SemanticTokenType;

pub const LEGEND_TYPE: &[SemanticTokenType] = &[
    SemanticTokenType::FUNCTION,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::STRING,
    SemanticTokenType::COMMENT,
    SemanticTokenType::NUMBER,
    SemanticTokenType::KEYWORD,
    SemanticTokenType::OPERATOR,
    SemanticTokenType::PARAMETER,
];

fn semantic_token_from_ast_function(
    semantic_tokens: &mut Vec<ImCompleteSemanticToken>,
    function: &Func,
    _func_name: &String,
) {
    function.args.iter().for_each(|(_, span)| {
        semantic_tokens.push(ImCompleteSemanticToken {
            start: span.start,
            length: span.len(),
            token_type: LEGEND_TYPE
                .iter()
                .position(|item| item == &SemanticTokenType::PARAMETER)
                .unwrap(),
        });
    });
    let (_, span) = &function.name;
    semantic_tokens.push(ImCompleteSemanticToken {
        start: span.start,
        length: span.len(),
        token_type: LEGEND_TYPE
            .iter()
            .position(|item| item == &SemanticTokenType::FUNCTION)
            .unwrap(),
    });
    semantic_token_from_expr(&function.body, semantic_tokens);
}

fn semantic_token_from_ast_import(_imp: &Import) {}

pub fn semantic_token_from_ast(
    ast: &HashMap<String, /*Func*/ TopLevelExpr>,
) -> Vec<ImCompleteSemanticToken> {
    let mut semantic_tokens: Vec<ImCompleteSemanticToken> = vec![];

    ast.iter().for_each(|(_func_name, top_level_expr)| {
        match top_level_expr {
            TopLevelExpr::Function(func) => {
                semantic_token_from_ast_function(&mut semantic_tokens, func, _func_name)
            }
            TopLevelExpr::Import(imp) => semantic_token_from_ast_import(imp),
        }

        /*function.args.iter().for_each(|(_, span)| {
            semantic_tokens.push(ImCompleteSemanticToken {
                start: span.start,
                length: span.len(),
                token_type: LEGEND_TYPE
                    .iter()
                    .position(|item| item == &SemanticTokenType::PARAMETER)
                    .unwrap(),
            });
        });
        let (_, span) = &function.name;
        semantic_tokens.push(ImCompleteSemanticToken {
            start: span.start,
            length: span.len(),
            token_type: LEGEND_TYPE
                .iter()
                .position(|item| item == &SemanticTokenType::FUNCTION)
                .unwrap(),
        });
        semantic_token_from_expr(&function.body, &mut semantic_tokens);*/
    });

    semantic_tokens
}

pub fn semantic_token_from_expr(
    expr: &Spanned<Expr>,
    semantic_tokens: &mut Vec<ImCompleteSemanticToken>,
) {
    match &expr.0 {
        Expr::Error => {}
        Expr::Value(_) => {}
        Expr::List(_) => {}
        Expr::Local((_name, span)) => {
            semantic_tokens.push(ImCompleteSemanticToken {
                start: span.start,
                length: span.len(),
                token_type: LEGEND_TYPE
                    .iter()
                    .position(|item| item == &SemanticTokenType::VARIABLE)
                    .unwrap(),
            });
        }
        Expr::Var(_, rhs, rest, name_span) => {
            semantic_tokens.push(ImCompleteSemanticToken {
                start: name_span.start,
                length: name_span.len(),
                token_type: LEGEND_TYPE
                    .iter()
                    .position(|item| item == &SemanticTokenType::VARIABLE)
                    .unwrap(),
            });
            semantic_token_from_expr(rhs, semantic_tokens);
            semantic_token_from_expr(rest, semantic_tokens);
        }
        Expr::Then(first, rest) => {
            semantic_token_from_expr(first, semantic_tokens);
            semantic_token_from_expr(rest, semantic_tokens);
        }
        Expr::Binary(lhs, _op, rhs) => {
            semantic_token_from_expr(lhs, semantic_tokens);
            semantic_token_from_expr(rhs, semantic_tokens);
        }
        Expr::Call(expr, params) => {
            semantic_token_from_expr(expr, semantic_tokens);
            params.0.iter().for_each(|p| {
                semantic_token_from_expr(p, semantic_tokens);
            });
        }
        Expr::If(test, consequent, alternative) => {
            semantic_token_from_expr(test, semantic_tokens);
            semantic_token_from_expr(consequent, semantic_tokens);
            semantic_token_from_expr(alternative, semantic_tokens);
        }
    }
}
