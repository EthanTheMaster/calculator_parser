
use crate::calculator::ast::{Values, Node};
use crate::calculator::{lexer, Tag, Nonterminal, ActionTable};
use crate::calculator;
use crate::parser::GrammarContext;
use std::cell::RefCell;

thread_local! {
    pub static CONTEXT_TABLE: RefCell<(GrammarContext<Nonterminal, Tag<'static>>, ActionTable)> = RefCell::new(calculator::generate_action_table());
}

#[allow(dead_code)]
fn compute_answer(input: &str) -> Values {
    let answer = RefCell::new(Values::Int(0));
    CONTEXT_TABLE.with(|context_table| {
        let grammar_context = &context_table.borrow().0;
        let action_table = &context_table.borrow().1;
        let tokens = lexer::lex(&input.chars().collect()).unwrap();
        let res = calculator::generate_parse_tree(&tokens, action_table, grammar_context);
        if let Node::Expr(expr_node) = res.unwrap() {
            *answer.borrow_mut() = expr_node.value
        } else {
            panic!("Malformed input");
        }
    });
    return answer.into_inner();
}

#[allow(dead_code)]
fn approx_eq(value1: &Values, value2: &Values) -> bool {
    match (value1, value2) {
        (Values::Int(n1), Values::Int(n2)) => {*n1 == *n2},
        (Values::Float(n1), Values::Float(n2)) => {
            // First check for raw equality. If it fails then fall back to epsilon method.
            if *n1 == *n2 {
                true
            } else {
                (*n1 - *n2).abs() < 1e-8
            }
        },
        _ => {false}
    }
}

#[test]
fn test_basic() {
    assert!(approx_eq(&compute_answer("2 + 2"), &Values::Int(4)));
    assert!(approx_eq(&compute_answer("-1 - 2"), &Values::Int(-3)));
    assert!(approx_eq(&compute_answer("3 / 2"), &Values::Int(1)));
    assert!(approx_eq(&compute_answer("3 * 2"), &Values::Int(6)));

    assert!(approx_eq(&compute_answer("2.0 + 2.0"), &Values::Float(4.0)));
    assert!(approx_eq(&compute_answer("-1.0 - 2.0"), &Values::Float(-3.0)));
    assert!(approx_eq(&compute_answer("3.0 / 2.0"), &Values::Float(1.5)));
    assert!(approx_eq(&compute_answer("3.0 * 2.0"), &Values::Float(6.0)));
}

#[test]
fn test_pemdas() {
    assert!(approx_eq(&compute_answer("2 + 2 * 3"), &Values::Int(8)));
    assert!(approx_eq(&compute_answer("2 + 1 - 4 / 2"), &Values::Int(1)));
    assert!(approx_eq(&compute_answer("2^2^3 * 3"), &Values::Int(768)));
    assert!(approx_eq(&compute_answer("2^2^(3 * 1)"), &Values::Int(256)));
    assert!(approx_eq(&compute_answer("1 - 3 * 2"), &Values::Int(-5)));
}

#[test]
fn test_side_by_side() {
    assert!(approx_eq(&compute_answer("(2+2)(1+1)"), &Values::Int(8)));
    assert!(approx_eq(&compute_answer("(2)^2(3)^3(4)^4"), &Values::Int(27648)));
    assert!(approx_eq(&compute_answer("(2+3)5(8/4)"), &Values::Int(50)));
    assert!(approx_eq(&compute_answer("2(9 + 10)sqrt(4.0)cos(3.14)"), &Values::Float(-75.999903611293)));
}

#[test]
fn test_stress() {
    assert!(approx_eq(&compute_answer("tan(1) - (2 + 3) - sin(log(2,3)^-5 - 3 * 2 / sqrt(4))"), &Values::Float(-3.203364173673404959405672272346027961372451593524401165323572710)));
    assert!(approx_eq(&compute_answer("-1-2--3+2+-5"), &Values::Int(-3)));
    assert!(approx_eq(&compute_answer("25 + -(9 + 1 / float(-3 ^ 3)) * 9"), &Values::Float(-55.66666666666666666666666666666666666666666666666666666666666666)));
    assert!(approx_eq(&compute_answer("22(3+5)/11sqrt(2^3)5^2log(3, 100^(1.0/2.0)--5+2+3*6.0/-7.0)"), &Values::Float(2748.8012226960535901380539085660574409717730060287020126533420624)));
    assert!(approx_eq(&compute_answer("1.0/5(1+1)sqrt(100)"), &Values::Float(4.0)));
}