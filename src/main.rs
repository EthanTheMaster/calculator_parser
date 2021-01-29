use calculator::lexer;

mod parser;
mod calculator;
mod tests;

use std::io::{stdin, stdout, Write};
use crate::calculator::ast::{Node, Values};

fn main() {
    println!("Generating grammar...");
    let (grammar_context, action_table) = calculator::generate_action_table();
    println!("Finished generating grammar! Ready to parse expressions!");
    let mut user_input = String::new();
    loop {
        let _ = stdout().flush();
        stdin().read_line(&mut user_input).unwrap();
        let tokens = lexer::lex(&user_input.chars().filter(|c| *c!='\n').collect());
        match tokens {
            Ok(tokens) => {
                let res = calculator::generate_parse_tree(&tokens, &action_table, &grammar_context);
                println!("{:#?}", res);
                match res {
                    Ok(Node::Expr(node)) => {
                        match node.value {
                            Values::Float(n) => {
                                println!("Answer: {}", n);
                            },
                            Values::Int(n) => {
                                println!("Answer: {}", n);
                            },
                        }
                    },
                    _ => {
                        println!("Failed to parse.");
                    }
                }
            },
            Err(msg) => {
                println!("{}", msg);
            },
        }
        // Clear input
        user_input = String::new();
    }
}
