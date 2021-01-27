use calculator::lexer;

mod parser;
mod calculator;

fn main() {
    let (grammar_context, action_table) = calculator::generate_action_table();
    let tokens = lexer::lex(&"tan(1) - (2 + 3) - sin(log(2,3)^-5 - 3 * 2 / sqrt(4))".chars().collect()).unwrap();
    let res = calculator::generate_parse_tree(&tokens, &action_table, &grammar_context);
    println!("{:#?}", res);
    println!("----------------------------");
    let tokens = lexer::lex(&"-1-2--3+2+-5".chars().collect()).unwrap();
    let res = calculator::generate_parse_tree(&tokens, &action_table, &grammar_context);
    println!("{:#?}", res);
    println!("----------------------------");
    let tokens = lexer::lex(&"5/(7^(1/2) - sqrt(7))".chars().collect()).unwrap();
    let res = calculator::generate_parse_tree(&tokens, &action_table, &grammar_context);
    println!("{:#?}", res);
    println!("----------------------------");
    let tokens = lexer::lex(&"25 + -(9 + 1 / float(-3 ^ 3)) * 9".chars().collect()).unwrap();
    let res = calculator::generate_parse_tree(&tokens, &action_table, &grammar_context);
    println!("{:#?}", res);
}
