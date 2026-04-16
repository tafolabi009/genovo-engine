//! Compiler for the Genovo scripting language.
//!
//! Transforms source text into bytecode through three phases:
//!
//! 1. **Lexing** — tokenizes source text into a stream of `Token`s.
//! 2. **Parsing** — recursive-descent parser with Pratt precedence for
//!    expressions, producing an implicit AST that is compiled on the fly.
//! 3. **Code generation** — emits `OpCode` instructions into `Chunk`s.
//!
//! The language is a simple, imperative scripting language with C-like syntax:
//!
//! ```text
//! let speed = 5.0
//! let health = 100
//!
//! fn update(dt) {
//!     if health > 0 {
//!         let move_x = speed * dt
//!         print(move_x)
//!     }
//! }
//!
//! fn damage(amount) {
//!     health = health - amount
//!     if health <= 0 {
//!         print("dead!")
//!     }
//!     return health
//! }
//! ```

use super::{
    Chunk, FunctionId, OpCode, ScriptError, ScriptFunction, ScriptValue,
};

// ===========================================================================
// Tokens
// ===========================================================================

/// Token types produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // -- Literals --
    /// Integer literal (e.g. `42`).
    IntLiteral(i64),
    /// Float literal (e.g. `3.14`).
    FloatLiteral(f64),
    /// String literal (e.g. `"hello"`).
    StringLiteral(String),
    /// Identifier (e.g. `speed`, `health`).
    Identifier(String),

    // -- Single-character tokens --
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Dot,
    Semicolon,
    Colon,

    // -- One or two character tokens --
    Equal,
    EqualEqual,
    Bang,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    // -- Keywords --
    Let,
    Fn,
    If,
    Else,
    While,
    For,
    Return,
    True,
    False,
    Nil,
    And,
    Or,
    Not,
    Print,
    Break,
    Continue,

    // -- Special --
    Eof,
    Error(String),
}

/// A token with its source location.
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub line: u32,
    pub col: u32,
}

impl Token {
    fn new(kind: TokenKind, line: u32, col: u32) -> Self {
        Self { kind, line, col }
    }
}

// ===========================================================================
// Lexer
// ===========================================================================

/// Transforms source text into a sequence of tokens.
pub struct Lexer {
    /// Source code as a vector of characters for easy indexed access.
    chars: Vec<char>,
    /// Current character index.
    pos: usize,
    /// Current line number (1-based).
    line: u32,
    /// Current column number (1-based).
    col: u32,
}

impl Lexer {
    /// Create a new lexer for the given source text.
    pub fn new(source: &str) -> Self {
        Self {
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    /// Tokenize the entire source and return a vector of tokens.
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        tokens
    }

    /// Advance and return the next token.
    fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        if self.is_at_end() {
            return Token::new(TokenKind::Eof, self.line, self.col);
        }

        let start_line = self.line;
        let start_col = self.col;
        let c = self.advance();

        match c {
            '+' => Token::new(TokenKind::Plus, start_line, start_col),
            '-' => Token::new(TokenKind::Minus, start_line, start_col),
            '*' => Token::new(TokenKind::Star, start_line, start_col),
            '/' => Token::new(TokenKind::Slash, start_line, start_col),
            '%' => Token::new(TokenKind::Percent, start_line, start_col),
            '(' => Token::new(TokenKind::LeftParen, start_line, start_col),
            ')' => Token::new(TokenKind::RightParen, start_line, start_col),
            '{' => Token::new(TokenKind::LeftBrace, start_line, start_col),
            '}' => Token::new(TokenKind::RightBrace, start_line, start_col),
            '[' => Token::new(TokenKind::LeftBracket, start_line, start_col),
            ']' => Token::new(TokenKind::RightBracket, start_line, start_col),
            ',' => Token::new(TokenKind::Comma, start_line, start_col),
            '.' => Token::new(TokenKind::Dot, start_line, start_col),
            ';' => Token::new(TokenKind::Semicolon, start_line, start_col),
            ':' => Token::new(TokenKind::Colon, start_line, start_col),

            '=' => {
                if self.match_char('=') {
                    Token::new(TokenKind::EqualEqual, start_line, start_col)
                } else {
                    Token::new(TokenKind::Equal, start_line, start_col)
                }
            }
            '!' => {
                if self.match_char('=') {
                    Token::new(TokenKind::BangEqual, start_line, start_col)
                } else {
                    Token::new(TokenKind::Bang, start_line, start_col)
                }
            }
            '<' => {
                if self.match_char('=') {
                    Token::new(TokenKind::LessEqual, start_line, start_col)
                } else {
                    Token::new(TokenKind::Less, start_line, start_col)
                }
            }
            '>' => {
                if self.match_char('=') {
                    Token::new(TokenKind::GreaterEqual, start_line, start_col)
                } else {
                    Token::new(TokenKind::Greater, start_line, start_col)
                }
            }

            '"' => self.lex_string(start_line, start_col),

            c if c.is_ascii_digit() => self.lex_number(c, start_line, start_col),

            c if c.is_ascii_alphabetic() || c == '_' => {
                self.lex_identifier(c, start_line, start_col)
            }

            _ => Token::new(
                TokenKind::Error(format!("unexpected character '{c}'")),
                start_line,
                start_col,
            ),
        }
    }

    fn lex_string(&mut self, start_line: u32, start_col: u32) -> Token {
        let mut s = String::new();
        loop {
            if self.is_at_end() {
                return Token::new(
                    TokenKind::Error("unterminated string".into()),
                    start_line,
                    start_col,
                );
            }
            let c = self.advance();
            if c == '"' {
                break;
            }
            if c == '\\' {
                if self.is_at_end() {
                    return Token::new(
                        TokenKind::Error("unterminated escape in string".into()),
                        start_line,
                        start_col,
                    );
                }
                let esc = self.advance();
                match esc {
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    '\\' => s.push('\\'),
                    '"' => s.push('"'),
                    '0' => s.push('\0'),
                    _ => {
                        s.push('\\');
                        s.push(esc);
                    }
                }
            } else {
                if c == '\n' {
                    self.line += 1;
                    self.col = 1;
                }
                s.push(c);
            }
        }
        Token::new(TokenKind::StringLiteral(s), start_line, start_col)
    }

    fn lex_number(&mut self, first: char, start_line: u32, start_col: u32) -> Token {
        let mut s = String::new();
        s.push(first);
        let mut is_float = false;

        while !self.is_at_end() {
            let c = self.peek_char();
            if c.is_ascii_digit() {
                s.push(self.advance());
            } else if c == '.' && !is_float {
                // Check the character after the dot — if it's a digit, consume it.
                if self.peek_next().map_or(false, |nc| nc.is_ascii_digit()) {
                    is_float = true;
                    s.push(self.advance()); // consume '.'
                } else {
                    break;
                }
            } else if c == '_' {
                // Allow underscores in numbers (1_000_000).
                self.advance();
            } else {
                break;
            }
        }

        if is_float {
            match s.parse::<f64>() {
                Ok(v) => Token::new(TokenKind::FloatLiteral(v), start_line, start_col),
                Err(e) => Token::new(
                    TokenKind::Error(format!("invalid float: {e}")),
                    start_line,
                    start_col,
                ),
            }
        } else {
            match s.parse::<i64>() {
                Ok(v) => Token::new(TokenKind::IntLiteral(v), start_line, start_col),
                Err(e) => Token::new(
                    TokenKind::Error(format!("invalid int: {e}")),
                    start_line,
                    start_col,
                ),
            }
        }
    }

    fn lex_identifier(&mut self, first: char, start_line: u32, start_col: u32) -> Token {
        let mut s = String::new();
        s.push(first);
        while !self.is_at_end() {
            let c = self.peek_char();
            if c.is_ascii_alphanumeric() || c == '_' {
                s.push(self.advance());
            } else {
                break;
            }
        }

        let kind = match s.as_str() {
            "let" => TokenKind::Let,
            "fn" => TokenKind::Fn,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "return" => TokenKind::Return,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "nil" => TokenKind::Nil,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            "print" => TokenKind::Print,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            _ => TokenKind::Identifier(s),
        };

        Token::new(kind, start_line, start_col)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            if self.is_at_end() {
                return;
            }
            let c = self.peek_char();
            match c {
                ' ' | '\t' | '\r' => {
                    self.advance();
                }
                '\n' => {
                    self.advance();
                    self.line += 1;
                    self.col = 1;
                }
                '/' => {
                    if self.peek_next() == Some('/') {
                        // Line comment — skip to end of line.
                        while !self.is_at_end() && self.peek_char() != '\n' {
                            self.advance();
                        }
                    } else if self.peek_next() == Some('*') {
                        // Block comment — skip to */.
                        self.advance(); // consume '/'
                        self.advance(); // consume '*'
                        let mut depth = 1;
                        while !self.is_at_end() && depth > 0 {
                            let ch = self.advance();
                            if ch == '\n' {
                                self.line += 1;
                                self.col = 1;
                            } else if ch == '/' && self.peek_char() == '*' {
                                self.advance();
                                depth += 1;
                            } else if ch == '*' && self.peek_char() == '/' {
                                self.advance();
                                depth -= 1;
                            }
                        }
                    } else {
                        return;
                    }
                }
                _ => return,
            }
        }
    }

    #[inline]
    fn is_at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    #[inline]
    fn peek_char(&self) -> char {
        self.chars[self.pos]
    }

    #[inline]
    fn peek_next(&self) -> Option<char> {
        if self.pos + 1 < self.chars.len() {
            Some(self.chars[self.pos + 1])
        } else {
            None
        }
    }

    fn advance(&mut self) -> char {
        let c = self.chars[self.pos];
        self.pos += 1;
        self.col += 1;
        c
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() || self.peek_char() != expected {
            false
        } else {
            self.advance();
            true
        }
    }
}

// ===========================================================================
// Precedence (for Pratt parsing)
// ===========================================================================

/// Operator precedence levels, lowest to highest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum Precedence {
    None = 0,
    Assignment = 1, // =
    Or = 2,         // or
    And = 3,        // and
    Equality = 4,   // == !=
    Comparison = 5, // < > <= >=
    Term = 6,       // + -
    Factor = 7,     // * / %
    Unary = 8,      // - not !
    Call = 9,       // . () []
    Primary = 10,
}

impl Precedence {
    fn next(self) -> Precedence {
        match self {
            Precedence::None => Precedence::Assignment,
            Precedence::Assignment => Precedence::Or,
            Precedence::Or => Precedence::And,
            Precedence::And => Precedence::Equality,
            Precedence::Equality => Precedence::Comparison,
            Precedence::Comparison => Precedence::Term,
            Precedence::Term => Precedence::Factor,
            Precedence::Factor => Precedence::Unary,
            Precedence::Unary => Precedence::Call,
            Precedence::Call => Precedence::Primary,
            Precedence::Primary => Precedence::Primary,
        }
    }
}

/// Returns (prefix precedence, infix precedence) for a token kind.
fn get_precedence(kind: &TokenKind) -> Precedence {
    match kind {
        TokenKind::Or => Precedence::Or,
        TokenKind::And => Precedence::And,
        TokenKind::EqualEqual | TokenKind::BangEqual => Precedence::Equality,
        TokenKind::Less
        | TokenKind::LessEqual
        | TokenKind::Greater
        | TokenKind::GreaterEqual => Precedence::Comparison,
        TokenKind::Plus | TokenKind::Minus => Precedence::Term,
        TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Precedence::Factor,
        TokenKind::LeftParen | TokenKind::LeftBracket | TokenKind::Dot => Precedence::Call,
        _ => Precedence::None,
    }
}

// ===========================================================================
// Local variable resolution
// ===========================================================================

/// A local variable in the current scope.
#[derive(Debug, Clone)]
struct Local {
    /// The variable name.
    name: String,
    /// The scope depth at which this variable was declared.
    depth: u32,
    /// The stack slot index.
    slot: u16,
}

// ===========================================================================
// Compiler
// ===========================================================================

/// Result of a successful compilation.
pub struct CompileResult {
    /// The top-level script function (always function index 0).
    pub script: ScriptFunction,
    /// Additional functions defined in the script.
    pub functions: Vec<ScriptFunction>,
}

/// The compiler: parses tokens and emits bytecode.
///
/// Uses a single-pass, recursive-descent approach with Pratt parsing for
/// expression precedence. Variables are resolved at compile time to either
/// local slots or global names.
pub struct Compiler {
    /// Token stream from the lexer.
    tokens: Vec<Token>,
    /// Current position in the token stream.
    current: usize,
    /// The function currently being compiled.
    chunk: Chunk,
    /// Local variables in the current scope.
    locals: Vec<Local>,
    /// Current scope depth (0 = global/script level).
    scope_depth: u32,
    /// Next available local slot.
    next_slot: u16,
    /// Functions compiled so far (sub-functions of the script).
    compiled_functions: Vec<ScriptFunction>,
    /// The name of the script being compiled (for error messages).
    script_name: String,
    /// Current line (for error reporting).
    current_line: u32,
    /// Whether the compiler has encountered an error.
    had_error: bool,
    /// Collected error messages.
    errors: Vec<String>,
    /// The set of global variable names encountered in the script.
    /// Used so that we emit GetGlobal/SetGlobal for top-level `let` bindings
    /// and for variables not resolved as locals.
    global_names: Vec<String>,
    /// Track the names of native functions that should be called with CallNative.
    native_names: Vec<String>,
}

impl Compiler {
    /// Create a new compiler for the given source code.
    pub fn new(source: &str, name: &str) -> Self {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();

        Self {
            tokens,
            current: 0,
            chunk: Chunk::new(),
            locals: Vec::new(),
            scope_depth: 0,
            next_slot: 0,
            compiled_functions: Vec::new(),
            script_name: name.to_string(),
            current_line: 1,
            had_error: false,
            errors: Vec::new(),
            global_names: Vec::new(),
            native_names: Vec::new(),
        }
    }

    /// Register a name as a known native function.
    pub fn register_native_name(&mut self, name: &str) {
        self.native_names.push(name.to_string());
    }

    /// Compile the source and return the result.
    pub fn compile(&mut self) -> Result<CompileResult, ScriptError> {
        while !self.check(TokenKind::Eof) {
            self.declaration()?;
        }

        // Implicit return Nil at end of script.
        self.emit_op(OpCode::PushNil);
        self.emit_op(OpCode::Return);

        if self.had_error {
            return Err(ScriptError::CompileError(self.errors.join("\n")));
        }

        let mut script = ScriptFunction::new(
            format!("<script:{}>", self.script_name),
            0,
        );
        script.chunk = std::mem::replace(&mut self.chunk, Chunk::new());
        script.local_count = self.next_slot as u8;

        Ok(CompileResult {
            script,
            functions: std::mem::take(&mut self.compiled_functions),
        })
    }

    // -- Token helpers ------------------------------------------------------

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.tokens[self.current].kind
    }

    #[allow(dead_code)]
    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn advance_token(&mut self) -> &Token {
        let tok = &self.tokens[self.current];
        if tok.kind != TokenKind::Eof {
            self.current += 1;
        }
        self.current_line = self.tokens[self.current.min(self.tokens.len() - 1)].line;
        &self.tokens[self.current - 1]
    }

    fn check(&self, kind: TokenKind) -> bool {
        std::mem::discriminant(&self.tokens[self.current].kind)
            == std::mem::discriminant(&kind)
    }

    #[allow(dead_code)]
    fn check_identifier(&self) -> bool {
        matches!(self.tokens[self.current].kind, TokenKind::Identifier(_))
    }

    fn match_token(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance_token();
            true
        } else {
            false
        }
    }

    fn consume(&mut self, kind: TokenKind, msg: &str) -> Result<(), ScriptError> {
        if self.check(kind) {
            self.advance_token();
            Ok(())
        } else {
            let actual = format!("{:?}", self.peek().kind);
            self.error(&format!("{msg}, got {actual}"))
        }
    }

    fn error(&mut self, msg: &str) -> Result<(), ScriptError> {
        let line = self.peek().line;
        let full = format!(
            "[{}:{}] error: {msg}",
            self.script_name, line
        );
        self.had_error = true;
        self.errors.push(full.clone());
        Err(ScriptError::CompileError(full))
    }

    #[allow(dead_code)]
    fn error_at_previous(&mut self, msg: &str) -> Result<(), ScriptError> {
        let line = self.previous().line;
        let full = format!(
            "[{}:{}] error: {msg}",
            self.script_name, line
        );
        self.had_error = true;
        self.errors.push(full.clone());
        Err(ScriptError::CompileError(full))
    }

    // -- Emit helpers -------------------------------------------------------

    fn emit_op(&mut self, op: OpCode) {
        self.chunk.emit_op(op, self.current_line);
    }

    fn emit_op_u16(&mut self, op: OpCode, operand: u16) {
        self.chunk.emit_op_u16(op, operand, self.current_line);
    }

    fn emit_constant(&mut self, value: ScriptValue) {
        self.chunk.emit_constant(value, self.current_line);
    }

    fn emit_jump(&mut self, op: OpCode) -> usize {
        self.chunk.emit_jump(op, self.current_line)
    }

    fn patch_jump(&mut self, offset: usize) -> Result<(), ScriptError> {
        self.chunk.patch_jump(offset)
    }

    fn emit_loop(&mut self, loop_start: usize) -> Result<(), ScriptError> {
        self.chunk.emit_loop(loop_start, self.current_line)
    }

    // -- Scope management ---------------------------------------------------

    fn begin_scope(&mut self) {
        self.scope_depth += 1;
    }

    fn end_scope(&mut self) {
        self.scope_depth -= 1;
        // Pop locals that are going out of scope.
        while let Some(local) = self.locals.last() {
            if local.depth <= self.scope_depth {
                break;
            }
            self.emit_op(OpCode::Pop);
            self.locals.pop();
        }
    }

    fn add_local(&mut self, name: &str) -> u16 {
        let slot = self.next_slot;
        self.locals.push(Local {
            name: name.to_string(),
            depth: self.scope_depth,
            slot,
        });
        self.next_slot += 1;
        slot
    }

    fn resolve_local(&self, name: &str) -> Option<u16> {
        for local in self.locals.iter().rev() {
            if local.name == name {
                return Some(local.slot);
            }
        }
        None
    }

    fn is_global_scope(&self) -> bool {
        self.scope_depth == 0
    }

    fn is_native(&self, name: &str) -> bool {
        self.native_names.iter().any(|n| n == name)
    }

    // -- Declarations -------------------------------------------------------

    fn declaration(&mut self) -> Result<(), ScriptError> {
        if self.check(TokenKind::Let) {
            self.let_declaration()
        } else if self.check(TokenKind::Fn) {
            self.fn_declaration()
        } else {
            self.statement()
        }
    }

    fn let_declaration(&mut self) -> Result<(), ScriptError> {
        self.advance_token(); // consume 'let'

        let name = match self.peek_kind().clone() {
            TokenKind::Identifier(s) => s,
            _ => {
                return self.error("expected variable name after 'let'");
            }
        };
        self.advance_token();

        // Optional initializer.
        if self.match_token(TokenKind::Equal) {
            self.expression()?;
        } else {
            self.emit_op(OpCode::PushNil);
        }

        // Optional semicolon.
        self.match_token(TokenKind::Semicolon);

        if self.is_global_scope() {
            // Global variable.
            let name_idx = self
                .chunk
                .add_constant(ScriptValue::from_string(&name));
            self.emit_op_u16(OpCode::SetGlobal, name_idx);
            self.emit_op(OpCode::Pop);
            self.global_names.push(name);
        } else {
            // Local variable — the value is already on the stack.
            let _slot = self.add_local(&name);
        }

        Ok(())
    }

    fn fn_declaration(&mut self) -> Result<(), ScriptError> {
        self.advance_token(); // consume 'fn'

        let name = match self.peek_kind().clone() {
            TokenKind::Identifier(s) => s,
            _ => {
                return self.error("expected function name after 'fn'");
            }
        };
        self.advance_token();

        self.consume(TokenKind::LeftParen, "expected '(' after function name")?;

        // Parse parameter names.
        let mut params: Vec<String> = Vec::new();
        if !self.check(TokenKind::RightParen) {
            loop {
                let param = match self.peek_kind().clone() {
                    TokenKind::Identifier(s) => s,
                    _ => {
                        return self.error("expected parameter name");
                    }
                };
                self.advance_token();
                params.push(param);
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenKind::RightParen, "expected ')' after parameters")?;

        let arity = params.len() as u8;

        // Save current compiler state.
        let saved_chunk = std::mem::replace(&mut self.chunk, Chunk::new());
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_scope = self.scope_depth;
        let saved_next_slot = self.next_slot;

        // Start a new scope for the function body.
        self.scope_depth = 1; // function body is at depth 1
        self.next_slot = 0;

        // Slot 0 is reserved for the function reference itself.
        self.add_local(&name);

        // Add parameter locals.
        for param in &params {
            self.add_local(param);
        }

        // Parse body.
        self.consume(TokenKind::LeftBrace, "expected '{' before function body")?;
        self.block()?;

        // Implicit return nil.
        self.emit_op(OpCode::PushNil);
        self.emit_op(OpCode::Return);

        let func_chunk = std::mem::replace(&mut self.chunk, saved_chunk);
        let func_local_count = self.next_slot;

        // Restore compiler state.
        self.locals = saved_locals;
        self.scope_depth = saved_scope;
        self.next_slot = saved_next_slot;

        let mut func = ScriptFunction::new(&name, arity);
        func.chunk = func_chunk;
        func.local_count = func_local_count as u8;

        // The function index will be: 1 + compiled_functions.len() (because
        // index 0 is the top-level script).
        let func_id = 1 + self.compiled_functions.len() as u32;
        self.compiled_functions.push(func);

        // In the enclosing scope, store the function value as a global.
        let name_idx = self
            .chunk
            .add_constant(ScriptValue::from_string(&name));
        self.emit_constant(ScriptValue::Function(FunctionId(func_id)));
        self.emit_op_u16(OpCode::SetGlobal, name_idx);
        self.emit_op(OpCode::Pop);

        Ok(())
    }

    // -- Statements ---------------------------------------------------------

    fn statement(&mut self) -> Result<(), ScriptError> {
        if self.check(TokenKind::If) {
            self.if_statement()
        } else if self.check(TokenKind::While) {
            self.while_statement()
        } else if self.check(TokenKind::For) {
            self.for_statement()
        } else if self.check(TokenKind::Return) {
            self.return_statement()
        } else if self.check(TokenKind::Print) {
            self.print_statement()
        } else if self.check(TokenKind::LeftBrace) {
            self.advance_token();
            self.begin_scope();
            self.block()?;
            self.end_scope();
            Ok(())
        } else {
            self.expression_statement()
        }
    }

    fn block(&mut self) -> Result<(), ScriptError> {
        while !self.check(TokenKind::RightBrace) && !self.check(TokenKind::Eof) {
            self.declaration()?;
        }
        self.consume(TokenKind::RightBrace, "expected '}' after block")?;
        Ok(())
    }

    fn if_statement(&mut self) -> Result<(), ScriptError> {
        self.advance_token(); // consume 'if'

        // Condition.
        self.expression()?;

        let else_jump = self.emit_jump(OpCode::JumpIfFalse);

        // Then branch.
        self.consume(TokenKind::LeftBrace, "expected '{' after if condition")?;
        self.begin_scope();
        self.block()?;
        self.end_scope();

        if self.check(TokenKind::Else) {
            let end_jump = self.emit_jump(OpCode::Jump);
            self.patch_jump(else_jump)?;

            self.advance_token(); // consume 'else'

            if self.check(TokenKind::If) {
                // else if
                self.if_statement()?;
            } else {
                self.consume(TokenKind::LeftBrace, "expected '{' after 'else'")?;
                self.begin_scope();
                self.block()?;
                self.end_scope();
            }

            self.patch_jump(end_jump)?;
        } else {
            self.patch_jump(else_jump)?;
        }

        Ok(())
    }

    fn while_statement(&mut self) -> Result<(), ScriptError> {
        self.advance_token(); // consume 'while'

        let loop_start = self.chunk.len();

        // Condition.
        self.expression()?;

        let exit_jump = self.emit_jump(OpCode::JumpIfFalse);

        // Body.
        self.consume(TokenKind::LeftBrace, "expected '{' after while condition")?;
        self.begin_scope();
        self.block()?;
        self.end_scope();

        self.emit_loop(loop_start)?;
        self.patch_jump(exit_jump)?;

        Ok(())
    }

    fn for_statement(&mut self) -> Result<(), ScriptError> {
        self.advance_token(); // consume 'for'

        self.begin_scope();

        // Initializer: `let i = 0` or just an expression.
        if self.check(TokenKind::Let) {
            self.let_declaration()?;
        } else if !self.match_token(TokenKind::Semicolon) {
            self.expression_statement()?;
        }

        let loop_start = self.chunk.len();

        // Condition (optional, defaults to true).
        let exit_jump;
        if !self.match_token(TokenKind::Semicolon) {
            self.expression()?;
            self.consume(TokenKind::Semicolon, "expected ';' after for condition")?;
            exit_jump = Some(self.emit_jump(OpCode::JumpIfFalse));
        } else {
            exit_jump = None;
        }

        // Increment (optional).
        // We compile the increment but jump over it initially, then loop back to it.
        let has_increment = !self.check(TokenKind::LeftBrace);
        let body_jump;
        let increment_start;
        if has_increment {
            body_jump = Some(self.emit_jump(OpCode::Jump));
            increment_start = self.chunk.len();
            self.expression()?;
            self.emit_op(OpCode::Pop);
            self.emit_loop(loop_start)?;
            if let Some(bj) = body_jump {
                self.patch_jump(bj)?;
            }
        } else {
            increment_start = loop_start;
        }

        // Body.
        self.consume(TokenKind::LeftBrace, "expected '{' after for clauses")?;
        self.begin_scope();
        self.block()?;
        self.end_scope();

        self.emit_loop(increment_start)?;

        if let Some(ej) = exit_jump {
            self.patch_jump(ej)?;
        }

        self.end_scope();

        Ok(())
    }

    fn return_statement(&mut self) -> Result<(), ScriptError> {
        self.advance_token(); // consume 'return'

        if self.check(TokenKind::RightBrace)
            || self.check(TokenKind::Eof)
            || self.check(TokenKind::Semicolon)
        {
            self.emit_op(OpCode::PushNil);
        } else {
            self.expression()?;
        }

        self.emit_op(OpCode::Return);
        self.match_token(TokenKind::Semicolon);
        Ok(())
    }

    fn print_statement(&mut self) -> Result<(), ScriptError> {
        self.advance_token(); // consume 'print'

        // print can be used as `print(expr)` or `print expr`.
        if self.match_token(TokenKind::LeftParen) {
            self.expression()?;
            self.consume(TokenKind::RightParen, "expected ')' after print argument")?;
        } else {
            self.expression()?;
        }

        self.emit_op(OpCode::Print);
        self.match_token(TokenKind::Semicolon);
        Ok(())
    }

    fn expression_statement(&mut self) -> Result<(), ScriptError> {
        self.expression()?;
        self.emit_op(OpCode::Pop);
        self.match_token(TokenKind::Semicolon);
        Ok(())
    }

    // -- Expressions (Pratt parsing) ----------------------------------------

    fn expression(&mut self) -> Result<(), ScriptError> {
        self.parse_precedence(Precedence::Assignment)
    }

    fn parse_precedence(&mut self, precedence: Precedence) -> Result<(), ScriptError> {
        // Prefix.
        let tok = self.advance_token().clone();
        self.prefix_rule(&tok)?;

        // Infix loop.
        while precedence <= get_precedence(self.peek_kind()) {
            let tok = self.advance_token().clone();
            self.infix_rule(&tok)?;
        }

        Ok(())
    }

    fn prefix_rule(&mut self, token: &Token) -> Result<(), ScriptError> {
        match &token.kind {
            TokenKind::IntLiteral(n) => {
                self.emit_constant(ScriptValue::Int(*n));
                Ok(())
            }
            TokenKind::FloatLiteral(f) => {
                self.emit_constant(ScriptValue::Float(*f));
                Ok(())
            }
            TokenKind::StringLiteral(s) => {
                self.emit_constant(ScriptValue::from_string(s));
                Ok(())
            }
            TokenKind::True => {
                self.emit_op(OpCode::PushTrue);
                Ok(())
            }
            TokenKind::False => {
                self.emit_op(OpCode::PushFalse);
                Ok(())
            }
            TokenKind::Nil => {
                self.emit_op(OpCode::PushNil);
                Ok(())
            }
            TokenKind::LeftParen => {
                self.expression()?;
                self.consume(TokenKind::RightParen, "expected ')' after expression")?;
                Ok(())
            }
            TokenKind::LeftBracket => {
                // Array literal: [expr, expr, ...]
                let mut count = 0u16;
                if !self.check(TokenKind::RightBracket) {
                    loop {
                        self.expression()?;
                        count += 1;
                        if !self.match_token(TokenKind::Comma) {
                            break;
                        }
                    }
                }
                self.consume(TokenKind::RightBracket, "expected ']' after array literal")?;
                self.emit_op_u16(OpCode::ArrayNew, count);
                Ok(())
            }
            TokenKind::Minus => {
                self.parse_precedence(Precedence::Unary)?;
                self.emit_op(OpCode::Neg);
                Ok(())
            }
            TokenKind::Not | TokenKind::Bang => {
                self.parse_precedence(Precedence::Unary)?;
                self.emit_op(OpCode::Not);
                Ok(())
            }
            TokenKind::Identifier(name) => {
                self.named_variable(name.clone())?;
                Ok(())
            }
            _ => {
                let msg = format!("unexpected token in expression: {:?}", token.kind);
                self.had_error = true;
                self.errors.push(format!(
                    "[{}:{}] error: {msg}",
                    self.script_name, token.line
                ));
                Err(ScriptError::CompileError(msg))
            }
        }
    }

    fn infix_rule(&mut self, token: &Token) -> Result<(), ScriptError> {
        match &token.kind {
            TokenKind::Plus => {
                self.parse_precedence(Precedence::Term.next())?;
                self.emit_op(OpCode::Add);
            }
            TokenKind::Minus => {
                self.parse_precedence(Precedence::Term.next())?;
                self.emit_op(OpCode::Sub);
            }
            TokenKind::Star => {
                self.parse_precedence(Precedence::Factor.next())?;
                self.emit_op(OpCode::Mul);
            }
            TokenKind::Slash => {
                self.parse_precedence(Precedence::Factor.next())?;
                self.emit_op(OpCode::Div);
            }
            TokenKind::Percent => {
                self.parse_precedence(Precedence::Factor.next())?;
                self.emit_op(OpCode::Mod);
            }
            TokenKind::EqualEqual => {
                self.parse_precedence(Precedence::Equality.next())?;
                self.emit_op(OpCode::Eq);
            }
            TokenKind::BangEqual => {
                self.parse_precedence(Precedence::Equality.next())?;
                self.emit_op(OpCode::Ne);
            }
            TokenKind::Less => {
                self.parse_precedence(Precedence::Comparison.next())?;
                self.emit_op(OpCode::Lt);
            }
            TokenKind::LessEqual => {
                self.parse_precedence(Precedence::Comparison.next())?;
                self.emit_op(OpCode::Le);
            }
            TokenKind::Greater => {
                self.parse_precedence(Precedence::Comparison.next())?;
                self.emit_op(OpCode::Gt);
            }
            TokenKind::GreaterEqual => {
                self.parse_precedence(Precedence::Comparison.next())?;
                self.emit_op(OpCode::Ge);
            }
            TokenKind::And => {
                self.parse_precedence(Precedence::And.next())?;
                self.emit_op(OpCode::And);
            }
            TokenKind::Or => {
                self.parse_precedence(Precedence::Or.next())?;
                self.emit_op(OpCode::Or);
            }
            TokenKind::LeftParen => {
                // Function call — the callee is already on the stack.
                self.finish_call()?;
            }
            TokenKind::LeftBracket => {
                // Array index: expr[idx]
                self.expression()?;
                self.consume(TokenKind::RightBracket, "expected ']' after index")?;
                self.emit_op(OpCode::ArrayGet);
            }
            TokenKind::Dot => {
                // Field access: expr.name
                let name = match self.peek_kind().clone() {
                    TokenKind::Identifier(s) => s,
                    _ => {
                        return self.error("expected field name after '.'");
                    }
                };
                self.advance_token();
                let name_idx = self
                    .chunk
                    .add_constant(ScriptValue::from_string(&name));
                self.emit_op_u16(OpCode::GetField, name_idx);
            }
            _ => {}
        }
        Ok(())
    }

    fn named_variable(&mut self, name: String) -> Result<(), ScriptError> {
        // Check if this is a function call (name followed by '(').
        // For native functions, use CallNative.
        if self.check(TokenKind::LeftParen) && self.is_native(&name) {
            self.advance_token(); // consume '('
            let arg_count = self.argument_list()?;
            let name_idx = self
                .chunk
                .add_constant(ScriptValue::from_string(&name));
            self.emit_op_u16(OpCode::CallNative, name_idx);
            self.chunk.code.push(arg_count);
            self.chunk.line_numbers.push(self.current_line);
            return Ok(());
        }

        // Check for assignment: name = expr
        if let Some(slot) = self.resolve_local(&name) {
            if self.check(TokenKind::Equal) && !self.check_two_equals() {
                self.advance_token(); // consume '='
                self.expression()?;
                self.emit_op_u16(OpCode::SetLocal, slot);
                // Leave value on stack (for expression result).
                return Ok(());
            }
            self.emit_op_u16(OpCode::GetLocal, slot);
        } else {
            // Global variable.
            if self.check(TokenKind::Equal) && !self.check_two_equals() {
                self.advance_token(); // consume '='
                self.expression()?;
                let name_idx = self
                    .chunk
                    .add_constant(ScriptValue::from_string(&name));
                self.emit_op_u16(OpCode::SetGlobal, name_idx);
                // Leave value on stack.
                return Ok(());
            }
            let name_idx = self
                .chunk
                .add_constant(ScriptValue::from_string(&name));
            self.emit_op_u16(OpCode::GetGlobal, name_idx);
        }

        Ok(())
    }

    /// Check if the next two chars are '==' (to distinguish from assignment).
    fn check_two_equals(&self) -> bool {
        if self.current + 1 < self.tokens.len() {
            matches!(self.tokens[self.current].kind, TokenKind::EqualEqual)
        } else {
            false
        }
    }

    fn finish_call(&mut self) -> Result<(), ScriptError> {
        let arg_count = self.argument_list()?;
        self.emit_op_u16(OpCode::Call, arg_count as u16);
        Ok(())
    }

    fn argument_list(&mut self) -> Result<u8, ScriptError> {
        let mut count: u16 = 0;
        if !self.check(TokenKind::RightParen) {
            loop {
                self.expression()?;
                count += 1;
                if count > 255 {
                    return Err(ScriptError::CompileError(
                        "cannot have more than 255 arguments".into(),
                    ));
                }
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenKind::RightParen, "expected ')' after arguments")?;
        Ok(count as u8)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::VM;

    // -----------------------------------------------------------------------
    // Lexer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lexer_basic_tokens() {
        let mut lexer = Lexer::new("+ - * / % ( ) { } [ ] , . ; :");
        let tokens = lexer.tokenize();
        let kinds: Vec<_> = tokens.iter().map(|t| &t.kind).collect();
        assert_eq!(kinds[0], &TokenKind::Plus);
        assert_eq!(kinds[1], &TokenKind::Minus);
        assert_eq!(kinds[2], &TokenKind::Star);
        assert_eq!(kinds[3], &TokenKind::Slash);
        assert_eq!(kinds[4], &TokenKind::Percent);
        assert_eq!(kinds[5], &TokenKind::LeftParen);
        assert_eq!(kinds[6], &TokenKind::RightParen);
        assert_eq!(kinds[7], &TokenKind::LeftBrace);
        assert_eq!(kinds[8], &TokenKind::RightBrace);
        assert_eq!(kinds[9], &TokenKind::LeftBracket);
        assert_eq!(kinds[10], &TokenKind::RightBracket);
        assert_eq!(kinds[11], &TokenKind::Comma);
        assert_eq!(kinds[12], &TokenKind::Dot);
        assert_eq!(kinds[13], &TokenKind::Semicolon);
        assert_eq!(kinds[14], &TokenKind::Colon);
        assert_eq!(kinds[15], &TokenKind::Eof);
    }

    #[test]
    fn test_lexer_two_char_tokens() {
        let mut lexer = Lexer::new("== != < <= > >=");
        let tokens = lexer.tokenize();
        let kinds: Vec<_> = tokens.iter().map(|t| &t.kind).collect();
        assert_eq!(kinds[0], &TokenKind::EqualEqual);
        assert_eq!(kinds[1], &TokenKind::BangEqual);
        assert_eq!(kinds[2], &TokenKind::Less);
        assert_eq!(kinds[3], &TokenKind::LessEqual);
        assert_eq!(kinds[4], &TokenKind::Greater);
        assert_eq!(kinds[5], &TokenKind::GreaterEqual);
    }

    #[test]
    fn test_lexer_numbers() {
        let mut lexer = Lexer::new("42 3.14 1_000");
        let tokens = lexer.tokenize();
        assert_eq!(tokens[0].kind, TokenKind::IntLiteral(42));
        assert_eq!(tokens[1].kind, TokenKind::FloatLiteral(3.14));
        assert_eq!(tokens[2].kind, TokenKind::IntLiteral(1000));
    }

    #[test]
    fn test_lexer_strings() {
        let mut lexer = Lexer::new(r#""hello" "world\n""#);
        let tokens = lexer.tokenize();
        assert_eq!(
            tokens[0].kind,
            TokenKind::StringLiteral("hello".to_string())
        );
        assert_eq!(
            tokens[1].kind,
            TokenKind::StringLiteral("world\n".to_string())
        );
    }

    #[test]
    fn test_lexer_keywords() {
        let mut lexer = Lexer::new("let fn if else while for return true false nil and or not print");
        let tokens = lexer.tokenize();
        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert_eq!(tokens[1].kind, TokenKind::Fn);
        assert_eq!(tokens[2].kind, TokenKind::If);
        assert_eq!(tokens[3].kind, TokenKind::Else);
        assert_eq!(tokens[4].kind, TokenKind::While);
        assert_eq!(tokens[5].kind, TokenKind::For);
        assert_eq!(tokens[6].kind, TokenKind::Return);
        assert_eq!(tokens[7].kind, TokenKind::True);
        assert_eq!(tokens[8].kind, TokenKind::False);
        assert_eq!(tokens[9].kind, TokenKind::Nil);
        assert_eq!(tokens[10].kind, TokenKind::And);
        assert_eq!(tokens[11].kind, TokenKind::Or);
        assert_eq!(tokens[12].kind, TokenKind::Not);
        assert_eq!(tokens[13].kind, TokenKind::Print);
    }

    #[test]
    fn test_lexer_identifiers() {
        let mut lexer = Lexer::new("speed health_max _temp x1");
        let tokens = lexer.tokenize();
        assert_eq!(
            tokens[0].kind,
            TokenKind::Identifier("speed".to_string())
        );
        assert_eq!(
            tokens[1].kind,
            TokenKind::Identifier("health_max".to_string())
        );
        assert_eq!(
            tokens[2].kind,
            TokenKind::Identifier("_temp".to_string())
        );
        assert_eq!(
            tokens[3].kind,
            TokenKind::Identifier("x1".to_string())
        );
    }

    #[test]
    fn test_lexer_comments() {
        let mut lexer = Lexer::new("42 // this is a comment\n+ 8");
        let tokens = lexer.tokenize();
        assert_eq!(tokens[0].kind, TokenKind::IntLiteral(42));
        assert_eq!(tokens[1].kind, TokenKind::Plus);
        assert_eq!(tokens[2].kind, TokenKind::IntLiteral(8));
    }

    #[test]
    fn test_lexer_block_comments() {
        let mut lexer = Lexer::new("42 /* block comment */ + 8");
        let tokens = lexer.tokenize();
        assert_eq!(tokens[0].kind, TokenKind::IntLiteral(42));
        assert_eq!(tokens[1].kind, TokenKind::Plus);
        assert_eq!(tokens[2].kind, TokenKind::IntLiteral(8));
    }

    #[test]
    fn test_lexer_line_tracking() {
        let mut lexer = Lexer::new("a\nb\nc");
        let tokens = lexer.tokenize();
        assert_eq!(tokens[0].line, 1);
        assert_eq!(tokens[1].line, 2);
        assert_eq!(tokens[2].line, 3);
    }

    // -----------------------------------------------------------------------
    // Compiler + VM integration tests
    // -----------------------------------------------------------------------

    fn compile_and_run(source: &str) -> Result<(ScriptValue, Vec<String>), ScriptError> {
        let mut compiler = Compiler::new(source, "test");
        let result = compiler.compile()?;

        let mut vm = VM::new();
        vm.load_function(result.script);
        for func in result.functions {
            vm.load_function(func);
        }

        let val = vm.run_script()?;
        let output = vm.output().to_vec();
        Ok((val, output))
    }

    fn compile_and_run_with_globals(
        source: &str,
        globals: Vec<(&str, ScriptValue)>,
    ) -> Result<(ScriptValue, Vec<String>), ScriptError> {
        let mut compiler = Compiler::new(source, "test");
        let result = compiler.compile()?;

        let mut vm = VM::new();
        for (name, val) in globals {
            vm.set_global(name, val);
        }
        vm.load_function(result.script);
        for func in result.functions {
            let name = func.name.clone();
            let id = vm.load_function(func);
            vm.set_global(name, ScriptValue::Function(FunctionId(id as u32)));
        }

        let val = vm.run_script()?;
        let output = vm.output().to_vec();
        Ok((val, output))
    }

    #[test]
    fn test_compile_int_literal() {
        let (val, _) = compile_and_run("return 42").unwrap();
        assert_eq!(val, ScriptValue::Int(42));
    }

    #[test]
    fn test_compile_float_literal() {
        let (val, _) = compile_and_run("return 3.14").unwrap();
        assert_eq!(val, ScriptValue::Float(3.14));
    }

    #[test]
    fn test_compile_string_literal() {
        let (val, _) = compile_and_run(r#"return "hello""#).unwrap();
        assert_eq!(val, ScriptValue::from_string("hello"));
    }

    #[test]
    fn test_compile_bool_literal() {
        let (val, _) = compile_and_run("return true").unwrap();
        assert_eq!(val, ScriptValue::Bool(true));
    }

    #[test]
    fn test_compile_nil_literal() {
        let (val, _) = compile_and_run("return nil").unwrap();
        assert_eq!(val, ScriptValue::Nil);
    }

    #[test]
    fn test_compile_arithmetic() {
        let (val, _) = compile_and_run("return 2 + 3 * 4").unwrap();
        // 2 + (3 * 4) = 14
        assert_eq!(val, ScriptValue::Int(14));
    }

    #[test]
    fn test_compile_arithmetic_precedence() {
        let (val, _) = compile_and_run("return (2 + 3) * 4").unwrap();
        assert_eq!(val, ScriptValue::Int(20));
    }

    #[test]
    fn test_compile_negation() {
        let (val, _) = compile_and_run("return -42").unwrap();
        assert_eq!(val, ScriptValue::Int(-42));
    }

    #[test]
    fn test_compile_comparison() {
        let (val, _) = compile_and_run("return 5 < 10").unwrap();
        assert_eq!(val, ScriptValue::Bool(true));

        let (val, _) = compile_and_run("return 10 <= 10").unwrap();
        assert_eq!(val, ScriptValue::Bool(true));

        let (val, _) = compile_and_run("return 10 > 5").unwrap();
        assert_eq!(val, ScriptValue::Bool(true));

        let (val, _) = compile_and_run("return 10 >= 11").unwrap();
        assert_eq!(val, ScriptValue::Bool(false));
    }

    #[test]
    fn test_compile_equality() {
        let (val, _) = compile_and_run("return 42 == 42").unwrap();
        assert_eq!(val, ScriptValue::Bool(true));

        let (val, _) = compile_and_run("return 42 != 43").unwrap();
        assert_eq!(val, ScriptValue::Bool(true));
    }

    #[test]
    fn test_compile_logic() {
        let (val, _) = compile_and_run("return not true").unwrap();
        assert_eq!(val, ScriptValue::Bool(false));

        let (val, _) = compile_and_run("return !false").unwrap();
        assert_eq!(val, ScriptValue::Bool(true));
    }

    #[test]
    fn test_compile_string_concat() {
        let (val, _) = compile_and_run(r#"return "hello " + "world""#).unwrap();
        assert_eq!(val, ScriptValue::from_string("hello world"));
    }

    #[test]
    fn test_compile_let_global() {
        let (_, output) = compile_and_run(
            r#"
            let x = 42
            print(x)
            "#,
        )
        .unwrap();
        assert_eq!(output, vec!["42"]);
    }

    #[test]
    fn test_compile_let_assignment() {
        let (_, output) = compile_and_run(
            r#"
            let x = 10
            x = x + 5
            print(x)
            "#,
        )
        .unwrap();
        assert_eq!(output, vec!["15"]);
    }

    #[test]
    fn test_compile_local_variables() {
        let (_, output) = compile_and_run(
            r#"
            let g = 100
            {
                let x = 10
                let y = 20
                print(x + y)
            }
            print(g)
            "#,
        )
        .unwrap();
        assert_eq!(output, vec!["30", "100"]);
    }

    #[test]
    fn test_compile_if_then() {
        let (_, output) = compile_and_run(
            r#"
            let x = 10
            if x > 5 {
                print("big")
            }
            "#,
        )
        .unwrap();
        assert_eq!(output, vec!["big"]);
    }

    #[test]
    fn test_compile_if_else() {
        let (_, output) = compile_and_run(
            r#"
            let x = 3
            if x > 5 {
                print("big")
            } else {
                print("small")
            }
            "#,
        )
        .unwrap();
        assert_eq!(output, vec!["small"]);
    }

    #[test]
    fn test_compile_if_else_if() {
        let (_, output) = compile_and_run(
            r#"
            let x = 5
            if x > 10 {
                print("big")
            } else if x > 3 {
                print("medium")
            } else {
                print("small")
            }
            "#,
        )
        .unwrap();
        assert_eq!(output, vec!["medium"]);
    }

    #[test]
    fn test_compile_while_loop() {
        let (_, output) = compile_and_run(
            r#"
            let i = 0
            let sum = 0
            while i < 5 {
                sum = sum + i
                i = i + 1
            }
            print(sum)
            "#,
        )
        .unwrap();
        // 0 + 1 + 2 + 3 + 4 = 10
        assert_eq!(output, vec!["10"]);
    }

    #[test]
    fn test_compile_function_def_and_call() {
        let (_, output) = compile_and_run_with_globals(
            r#"
            fn double(x) {
                return x * 2
            }
            print(double(21))
            "#,
            vec![],
        )
        .unwrap();
        assert_eq!(output, vec!["42"]);
    }

    #[test]
    fn test_compile_function_multiple_params() {
        let (_, output) = compile_and_run_with_globals(
            r#"
            fn add(a, b) {
                return a + b
            }
            print(add(10, 32))
            "#,
            vec![],
        )
        .unwrap();
        assert_eq!(output, vec!["42"]);
    }

    #[test]
    fn test_compile_function_no_return() {
        let (_, output) = compile_and_run_with_globals(
            r#"
            fn greet(name) {
                print("hello " + name)
            }
            greet("world")
            "#,
            vec![],
        )
        .unwrap();
        assert_eq!(output, vec!["hello world"]);
    }

    #[test]
    fn test_compile_function_accessing_global() {
        let (_, output) = compile_and_run_with_globals(
            r#"
            let speed = 5.0
            let health = 100

            fn damage(amount) {
                health = health - amount
                if health <= 0 {
                    print("dead!")
                }
                return health
            }

            let remaining = damage(60)
            print(remaining)
            remaining = damage(50)
            print(remaining)
            "#,
            vec![],
        )
        .unwrap();
        assert_eq!(output, vec!["40", "dead!", "-10"]);
    }

    #[test]
    fn test_compile_nested_function_calls() {
        let (_, output) = compile_and_run_with_globals(
            r#"
            fn square(x) {
                return x * x
            }
            fn sum_squares(a, b) {
                return square(a) + square(b)
            }
            print(sum_squares(3, 4))
            "#,
            vec![],
        )
        .unwrap();
        // 9 + 16 = 25
        assert_eq!(output, vec!["25"]);
    }

    #[test]
    fn test_compile_recursive_function() {
        let (_, output) = compile_and_run_with_globals(
            r#"
            fn factorial(n) {
                if n <= 1 {
                    return 1
                }
                return n * factorial(n - 1)
            }
            print(factorial(5))
            "#,
            vec![],
        )
        .unwrap();
        // 5! = 120
        assert_eq!(output, vec!["120"]);
    }

    #[test]
    fn test_compile_fibonacci() {
        let (_, output) = compile_and_run_with_globals(
            r#"
            fn fib(n) {
                if n <= 1 {
                    return n
                }
                return fib(n - 1) + fib(n - 2)
            }
            print(fib(10))
            "#,
            vec![],
        )
        .unwrap();
        // fib(10) = 55
        assert_eq!(output, vec!["55"]);
    }

    #[test]
    fn test_compile_array_literal() {
        let (val, _) = compile_and_run("return [1, 2, 3]").unwrap();
        assert_eq!(
            val,
            ScriptValue::Array(vec![
                ScriptValue::Int(1),
                ScriptValue::Int(2),
                ScriptValue::Int(3),
            ])
        );
    }

    #[test]
    fn test_compile_complex_script() {
        // The example script from the spec.
        let (_, output) = compile_and_run_with_globals(
            r#"
            let speed = 5.0
            let health = 100

            fn update(dt) {
                if health > 0 {
                    let move_x = speed * dt
                    print(move_x)
                }
            }

            fn damage(amount) {
                health = health - amount
                if health <= 0 {
                    print("dead!")
                }
                return health
            }

            update(0.016)
            damage(120)
            "#,
            vec![],
        )
        .unwrap();
        // speed(5.0) * dt(0.016) = 0.08
        // health(100) - 120 = -20, so prints "dead!"
        assert_eq!(output.len(), 2);
        assert!(output[0].starts_with("0.08"));
        assert_eq!(output[1], "dead!");
    }

    #[test]
    fn test_compile_modulo() {
        let (val, _) = compile_and_run("return 17 % 5").unwrap();
        assert_eq!(val, ScriptValue::Int(2));
    }

    #[test]
    fn test_compile_and_or() {
        let (val, _) = compile_and_run("return true and false").unwrap();
        assert_eq!(val, ScriptValue::Bool(false));

        let (val, _) = compile_and_run("return false or true").unwrap();
        assert_eq!(val, ScriptValue::Bool(true));
    }

    #[test]
    fn test_compile_nested_if() {
        let (_, output) = compile_and_run(
            r#"
            let x = 5
            if x > 0 {
                if x > 3 {
                    print("nested")
                }
            }
            "#,
        )
        .unwrap();
        assert_eq!(output, vec!["nested"]);
    }

    #[test]
    fn test_compile_while_with_local() {
        let (_, output) = compile_and_run(
            r#"
            let result = 1
            let i = 1
            while i <= 5 {
                result = result * i
                i = i + 1
            }
            print(result)
            "#,
        )
        .unwrap();
        // 5! = 120
        assert_eq!(output, vec!["120"]);
    }

    #[test]
    fn test_compile_comments() {
        let (val, _) = compile_and_run(
            r#"
            // This is a comment
            let x = 42 // inline comment
            /* block comment */
            return x
            "#,
        )
        .unwrap();
        assert_eq!(val, ScriptValue::Int(42));
    }

    #[test]
    fn test_compile_print_no_parens() {
        let (_, output) = compile_and_run(
            r#"
            print 42
            "#,
        )
        .unwrap();
        assert_eq!(output, vec!["42"]);
    }

    #[test]
    fn test_compile_multiple_returns() {
        let (val, _) = compile_and_run_with_globals(
            r#"
            fn abs(x) {
                if x < 0 {
                    return -x
                }
                return x
            }
            return abs(-7)
            "#,
            vec![],
        )
        .unwrap();
        assert_eq!(val, ScriptValue::Int(7));
    }

    #[test]
    fn test_compile_float_comparison() {
        let (val, _) = compile_and_run("return 3.14 > 2.71").unwrap();
        assert_eq!(val, ScriptValue::Bool(true));
    }

    #[test]
    fn test_compile_mixed_arithmetic() {
        let (val, _) = compile_and_run("return 10 + 2.5").unwrap();
        assert_eq!(val, ScriptValue::Float(12.5));
    }

    #[test]
    fn test_compile_empty_function() {
        let (_, output) = compile_and_run_with_globals(
            r#"
            fn noop() {
            }
            noop()
            print("done")
            "#,
            vec![],
        )
        .unwrap();
        assert_eq!(output, vec!["done"]);
    }

    #[test]
    fn test_compile_counter_pattern() {
        let (_, output) = compile_and_run_with_globals(
            r#"
            let count = 0

            fn increment() {
                count = count + 1
                return count
            }

            increment()
            increment()
            increment()
            print(count)
            "#,
            vec![],
        )
        .unwrap();
        assert_eq!(output, vec!["3"]);
    }

    #[test]
    fn test_native_function_via_compiler() {
        let mut compiler = Compiler::new(
            r#"
            let r = add_native(100, 200)
            return r
            "#,
            "test",
        );
        compiler.register_native_name("add_native");
        let result = compiler.compile().unwrap();

        let mut vm = VM::new();
        vm.register_native(
            "add_native",
            Box::new(|args| {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(ScriptValue::Int(a + b))
            }),
        );
        vm.load_function(result.script);
        for func in result.functions {
            vm.load_function(func);
        }

        let val = vm.run_script().unwrap();
        assert_eq!(val, ScriptValue::Int(300));
    }

    #[test]
    fn test_compile_escape_sequences() {
        let (val, _) = compile_and_run(r#"return "line1\nline2""#).unwrap();
        assert_eq!(val, ScriptValue::from_string("line1\nline2"));
    }

    #[test]
    fn test_lexer_nested_block_comments() {
        let mut lexer = Lexer::new("a /* outer /* inner */ still outer */ b");
        let tokens = lexer.tokenize();
        assert_eq!(
            tokens[0].kind,
            TokenKind::Identifier("a".to_string())
        );
        assert_eq!(
            tokens[1].kind,
            TokenKind::Identifier("b".to_string())
        );
    }
}
