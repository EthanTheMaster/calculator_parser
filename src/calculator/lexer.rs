use crate::parser::Normalize;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Token {
    Space,
    LParen,
    RParen,
    Add,
    Mult,
    Sub,
    Div,
    Exp,
    // String parameter holds the lexeme
    Int(String),
    Float(String),
    Id(String),
    Comma
}

impl Normalize for Token {
    // Allows us to check if two tokens are the same "type" by stripping out attribute information
    fn normalize(&self) -> Token {
        return match self {
            Token::Space
            | Token::LParen
            | Token::RParen
            | Token::Add
            | Token::Mult
            | Token::Sub
            | Token::Div
            | Token::Exp
            | Token::Comma => {
                self.clone()
            }
            Token::Int(_) => {Token::Int(String::new())},
            Token::Float(_) => {Token::Float(String::new())},
            Token::Id(_) => {Token::Id(String::new())},

        }
    }
}

pub fn get_token_from_first(first_char: char) -> Option<Token> {
    return match first_char {
        ' ' => Some(Token::Space),
        '(' => Some(Token::LParen),
        ')' => Some(Token::RParen),
        '+' => Some(Token::Add),
        '*' => Some(Token::Mult),
        '-' => Some(Token::Sub),
        '/' => Some(Token::Div),
        '^' => Some(Token::Exp),
        '0'..='9' => Some(Token::Int(String::from(first_char))),
        'a'..='z' | 'A'..='Z' => Some(Token::Id(String::from(first_char))),
        ',' => Some(Token::Comma),
        _ => None
    };
}

pub fn lex(string: &Vec<char>) -> Result<Vec<Token>, String> {
    if string.is_empty() {
        return Ok(vec![]);
    }

    let mut res = vec![];

    let mut stream = string.iter().enumerate();
    let mut current_token: Option<Token> = get_token_from_first(*stream.next().unwrap().1);
    for (idx, c) in stream {
        match current_token.take() {
            None => {
                // Error occurred at previous iteration
                return Err(format!("Failed to parse token at {}", idx - 1));
            },
            Some(token) => {
                match token {
                    Token::Space => {
                        // Don't add space to res. Continue lexing the next token.
                        current_token = get_token_from_first(*c);
                    },
                    Token::LParen
                    | Token::RParen
                    | Token::Add
                    | Token::Mult
                    | Token::Sub
                    | Token::Div
                    | Token::Exp
                    | Token:: Comma => {
                        // Completely matched token. Push it and start matching the next token.
                        res.push(token);
                        current_token = get_token_from_first(*c);
                    },
                    Token::Int(mut s) => {
                        match *c {
                            // If the character is a digit, append it to the current and update current token.
                            '0'..='9' => {
                                s.push(*c);
                                current_token = Some(Token::Int(s));
                            },
                            // Period signals that the number is no longer and integer.
                            '.' => {
                                // Convert into a float token.
                                s.push(*c);
                                current_token = Some(Token::Float(s));
                            }
                            // Integer token has ended. Start matching next token.
                            _ => {
                                res.push(Token::Int(s));
                                current_token = get_token_from_first(*c);
                            }
                        }
                    },
                    Token::Float(mut s) => {
                        match *c {
                            '0'..='9' => {
                                s.push(*c);
                                current_token = Some(Token::Float(s));
                            },
                            _ => {
                                res.push(Token::Float(s));
                                current_token = get_token_from_first(*c);
                            }
                        }
                    },
                    Token::Id(mut s) => {
                        match *c {
                            // Append alphanumeric characters to lexeme and update current_token
                            'a'..='z' | 'A'..='Z' | '_' | '0'..='9' => {
                                s.push(*c);
                                current_token = Some(Token::Id(s));
                            },
                            _ => {
                                res.push(Token::Id(s));
                                current_token = get_token_from_first(*c);
                            }
                        }
                    },
                }
            },
        }
    }

    // Token being matched has ended. Add it.
    res.push(current_token.unwrap());

    return Ok(res);
}