use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use ropey::Rope;
use dashmap::DashMap;
use std::collections::HashMap;

use cpoint_lsp_server::semantic_token::LEGEND_TYPE;
use cpoint_lsp_server::parser::{parse, Func, ImCompleteSemanticToken};

#[derive(Debug)]
struct Backend {
    client: Client,
    document_map: DashMap<String, Rope>,
    ast_map: DashMap<String, HashMap<String, Func>>,
    semantic_token_map: DashMap<String, Vec<ImCompleteSemanticToken>>,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        //Ok(InitializeResult::default())
        Ok(InitializeResult {
            server_info: None,
            offset_encoding: None,
            capabilities: ServerCapabilities {
                inlay_hint_provider: Some(OneOf::Left(true)),
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![".".to_string()]),
                    work_done_progress_options: Default::default(),
                    all_commit_characters: None,
                    completion_item: None,
                }),
                workspace: Some(WorkspaceServerCapabilities {
                    workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                        supported: Some(true),
                        change_notifications: Some(OneOf::Left(true)),
                    }),
                    file_operations: None,
                }),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensRegistrationOptions(SemanticTokensRegistrationOptions {
                        text_document_registration_options: {
                            TextDocumentRegistrationOptions {
                                document_selector: Some(vec![DocumentFilter {
                                    language: Some("cpoint".to_string()),
                                    scheme: Some("file".to_string()),
                                    pattern: None,
                                }]),
                            }
                        },
                        semantic_tokens_options: SemanticTokensOptions {
                            work_done_progress_options: WorkDoneProgressOptions::default(),
                            legend: SemanticTokensLegend {
                                token_types: LEGEND_TYPE.into(),
                                token_modifiers: vec![],
                            },
                            range: Some(true),
                            full: Some(SemanticTokensFullOptions::Bool(true)),
                        },
                        static_registration_options: StaticRegistrationOptions::default(),
                    }
                    )
                ),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                rename_provider: Some(OneOf::Left(true)),
                ..ServerCapabilities::default()
            }
            
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "server initialized!")
            .await;
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "file opened!")
            .await;
        self.on_change(TextDocumentItem {
            uri: params.text_document.uri,
            text: params.text_document.text,
            version: params.text_document.version,
            language_id: params.text_document.language_id, 
        })
        .await
    }

    async fn did_change(&self, mut params: DidChangeTextDocumentParams){
        self.client
            .log_message(MessageType::INFO, "File changed")
            .await;
        self.on_change(TextDocumentItem {
            uri: params.text_document.uri,
            text: std::mem::take(&mut params.content_changes[0].text),
            version: params.text_document.version,
            language_id: "cpoint".to_string()
        })
        .await
    }

    async fn did_save(&self, _: DidSaveTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "file saved!")
            .await;
    }
    async fn did_close(&self, _: DidCloseTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "file closed!")
            .await;
    }
    /*async fn goto_definition(&self, params: GotoDefinitionParams) -> Result<Option<GotoDefinitionResponse>> {
        let definition = async {
            let uri = params.text_document_position_params.text_document.uri;
            self.client
                .log_message(MessageType::INFO, "goto_definition")
                .await;
            let start_position = Position { line: 0, character: 0 };
            let end_position = Position { line: 0, character: 0 };
            let range = Range::new(start_position, end_position);
            Some(GotoDefinitionResponse::Scalar(Location::new(uri, range)))
        }.await;
        Ok(definition)
    }*/

    async fn semantic_tokens_range(&self, params: SemanticTokensRangeParams) -> Result<Option<SemanticTokensRangeResult>> {
        let _uri = params.text_document.uri.to_string();
        Ok(None)
    }

    async fn semantic_tokens_full(&self, _params: SemanticTokensParams) -> Result<Option<SemanticTokensResult>> {
        Ok(None)
    }

    async fn inlay_hint(&self, _params: tower_lsp::lsp_types::InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        self.client
            .log_message(MessageType::INFO, "inlay hint")
            .await;
        Ok(None)
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        self.client
            .log_message(MessageType::INFO, "completion")
            .await;
        let _uri = params.text_document_position.text_document.uri;
        let _position = params.text_document_position.position;
        Ok(None)
    }
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

impl Backend {
    async fn on_change(&self, params: TextDocumentItem){
        let rope = ropey::Rope::from_str(&params.text);
        self.document_map
            .insert(params.uri.to_string(), rope.clone());
        let (ast, _errors, _semantic_tokens) = parse(&params.text);
        if let Some(ast) = ast {
            self.ast_map.insert(params.uri.to_string(), ast);
        }
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    /*let (service, socket) = LspService::new(|client| Backend { client });
    Server::new(stdin, stdout, socket).serve(service).await;*/
    let (service, socket) = LspService::build(|client| Backend {
        client,
        document_map: DashMap::new(),
        ast_map: DashMap::new(),
        semantic_token_map: DashMap::new()
    }).finish();
    Server::new(stdin, stdout, socket).serve(service).await;


}