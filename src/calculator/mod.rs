use crate::parser::{ProductionRule, Symbol, LRAutomata, Action, ActionQuery, GrammarContext};
use lexer::{Token};
use std::collections::HashMap;
use std::iter::FromIterator;
use std::cmp::Ordering;
use crate::calculator::ast::{Node, Values, Expr, UnaryOp, BinOp};

pub mod ast;
pub mod lexer;

type Nonterminal = &'static str;

const START: Nonterminal = "Start";
const EXPR: Nonterminal = "Expression";
const PARAM: Nonterminal = "Parameter";
const PARAM_R: Nonterminal = "Remainder";

#[derive(Copy, Clone)]
enum Associativity {
    Left,
    Right,
}

#[derive(Copy, Clone)]
struct DisambiguationTag {
    associativity: Option<Associativity>,
    // the higher the value the more tightly the expression is bounded
    precedence: Option<usize>
}

impl DisambiguationTag {
    pub fn new(associativity: Option<Associativity>, precedence: Option<usize>) -> DisambiguationTag {
        DisambiguationTag {
            associativity,
            precedence
        }
    }
    
    pub fn empty() -> DisambiguationTag {
        DisambiguationTag {
            associativity: None,
            precedence: None
        }
    }
}

pub struct Tag<'a> {
    disambiguation: DisambiguationTag,
    // Syntax Directed Translation instructions
    sdt_instructions: &'a dyn Fn(Vec<Node>) -> Result<Node, String>,
}

impl<'a> Tag<'a> {
    fn new(disambiguation: DisambiguationTag, sdt_instructions: &'a dyn Fn(Vec<Node>) -> Result<Node, String>) -> Tag {
        Tag {
            disambiguation,
            sdt_instructions
        }
    }
}

type ActionTable = HashMap<ActionQuery<Nonterminal>, Action>;
pub fn generate_action_table<'a>() ->  (GrammarContext<Nonterminal, Tag<'a>>, ActionTable) {
    let op_disambiguation_map: HashMap<Token, DisambiguationTag> = HashMap::from_iter(vec![
        (Token::Add, DisambiguationTag::new(Some(Associativity::Left), Some(0))),
        (Token::Sub, DisambiguationTag::new(Some(Associativity::Left), Some(0))),
        (Token::Mult, DisambiguationTag::new(Some(Associativity::Left), Some(1))),
        (Token::Div, DisambiguationTag::new(Some(Associativity::Left), Some(1))),
        (Token::Exp, DisambiguationTag::new(Some(Associativity::Right), Some(2)))
    ].into_iter());
    // Define Grammar
    let production_rules: Vec<ProductionRule<Nonterminal, Tag>> = vec![
        //---Start Production---
        ProductionRule::new(
            START,
            vec![
                Symbol::Nonterminal(EXPR),
                Symbol::EndMarker
            ],
            Tag::new(
                DisambiguationTag::empty(),
                &move |mut children| {
                    let expr = children.pop().unwrap();
                    if let Node::Expr(_) = expr {
                        return Ok(expr)
                    } else {
                        panic!("Shift/Reduce parsing failure");
                    }
                }
            )
        ),
        // ---Expression Production---
        // Lone integer token
        ProductionRule::new(
            EXPR,
            vec![Symbol::Terminal(Token::Int(Default::default()))],
            Tag::new(
                DisambiguationTag::empty(),
                &move |mut children| {
                    let int_terminal = children.pop().unwrap();
                    if let Node::Terminal(Token::Int(int_string)) = int_terminal {
                        let int_parse = int_string.parse::<i64>().map_err(|_| format!("Failed to parse int."));
                        match int_parse {
                            Ok(int) => {
                                Expr::Number(Values::Int(int)).evaluate().map(|expr_node| Node::Expr(expr_node))
                            },
                            Err(msg) => {
                                Err(msg)
                            },
                        }

                    } else {
                        panic!("Shift/Reduce parsing failure");
                    }
                }
            )
        ),
        // Lone float token
        ProductionRule::new(
            EXPR,
            vec![Symbol::Terminal(Token::Float(Default::default()))],
            Tag::new(
                DisambiguationTag::empty(),
                 &move |mut children| {
                     let float_terminal = children.pop().unwrap();
                     if let Node::Terminal(Token::Float(float_string)) = float_terminal {
                         let float_parse = float_string.parse::<f64>().map_err(|_| format!("Failed to parse float."));
                         match float_parse {
                             Ok(float) => {
                                 Expr::Number(Values::Float(float)).evaluate().map(|expr_node| Node::Expr(expr_node))
                             },
                             Err(msg) => {
                                 Err(msg)
                             },
                         }

                     } else {
                         panic!("Shift/Reduce parsing failure");
                     }
                 }
            )
        ),
        // Parenthesized Expression
        ProductionRule::new(
            EXPR,
            vec![
                Symbol::Terminal(Token::LParen),
                Symbol::Nonterminal(EXPR),
                Symbol::Terminal(Token::RParen)
            ],
            Tag::new(
                DisambiguationTag::empty(),
                 &move |mut children| {
                     let _lparen = children.pop();
                     let expr = children.pop().unwrap();
                     let _rparen = children.pop();
                     Ok(expr)
                 }
            )
        ),
        // Negation of an Expression
        ProductionRule::new(
            EXPR,
            vec![
                Symbol::Terminal(Token::Sub),
                Symbol::Nonterminal(EXPR),
            ],
            Tag::new(
                DisambiguationTag::new(None, Some(999)),
                &move |mut children| {
                    let _neg_sign = children.pop();
                    let expr = children.pop().unwrap();
                    if let Node::Expr(expr_node) = expr {
                        let composite_expr = Expr::UnaryOp(UnaryOp::Neg, Box::new(expr_node));
                        composite_expr.evaluate().map(|expr_node| Node::Expr(expr_node))
                    } else {
                        panic!("Shift/Reduce parsing failure");
                    }
                }
            )
        ),
        // Addition of two expressions
        ProductionRule::new(
            EXPR,
            vec![
                Symbol::Nonterminal(EXPR),
                Symbol::Terminal(Token::Add),
                Symbol::Nonterminal(EXPR),
            ],
            Tag::new(
                op_disambiguation_map.get(&Token::Add).unwrap().clone(),
                &move |mut children| {
                    let lhs_expr = children.pop().unwrap();
                    let _add = children.pop();
                    let rhs_expr = children.pop().unwrap();
                    if let (Node::Expr(expr1_node), Node::Expr(expr2_node)) = (lhs_expr, rhs_expr) {
                        Expr::BinOp(BinOp::Add, Box::new(expr1_node), Box::new(expr2_node))
                            .evaluate()
                            .map(|expr_node| Node::Expr(expr_node))
                    } else {
                        panic!("Shift/Reduce parsing failure");
                    }
                }
            )
        ),
        // Subtraction of two expressions
        ProductionRule::new(
            EXPR,
            vec![
                Symbol::Nonterminal(EXPR),
                Symbol::Terminal(Token::Sub),
                Symbol::Nonterminal(EXPR),
            ],
            Tag::new(
                op_disambiguation_map.get(&Token::Sub).unwrap().clone(),
                &move |mut children| {
                    let lhs_expr = children.pop().unwrap();
                    let _sub = children.pop();
                    let rhs_expr = children.pop().unwrap();
                    if let (Node::Expr(expr1_node), Node::Expr(expr2_node)) = (lhs_expr, rhs_expr) {
                        Expr::BinOp(BinOp::Sub, Box::new(expr1_node), Box::new(expr2_node))
                            .evaluate()
                            .map(|expr_node| Node::Expr(expr_node))
                    } else {
                        panic!("Shift/Reduce parsing failure");
                    }
                }
            )
        ),
        // Multiplication of two expressions
        ProductionRule::new(
            EXPR,
            vec![
                Symbol::Nonterminal(EXPR),
                Symbol::Terminal(Token::Mult),
                Symbol::Nonterminal(EXPR),
            ],
            Tag::new(
                op_disambiguation_map.get(&Token::Mult).unwrap().clone(),
                &move |mut children| {
                    let lhs_expr = children.pop().unwrap();
                    let _mult = children.pop();
                    let rhs_expr = children.pop().unwrap();
                    if let (Node::Expr(expr1_node), Node::Expr(expr2_node)) = (lhs_expr, rhs_expr) {
                        Expr::BinOp(BinOp::Mult, Box::new(expr1_node), Box::new(expr2_node))
                            .evaluate()
                            .map(|expr_node| Node::Expr(expr_node))
                    } else {
                        panic!("Shift/Reduce parsing failure");
                    }
                }
            )
        ),
        // Division of two expressions
        ProductionRule::new(
            EXPR,
            vec![
                Symbol::Nonterminal(EXPR),
                Symbol::Terminal(Token::Div),
                Symbol::Nonterminal(EXPR),
            ],
            Tag::new(
                op_disambiguation_map.get(&Token::Div).unwrap().clone(),
                &move |mut children| {
                    let lhs_expr = children.pop().unwrap();
                    let _div = children.pop();
                    let rhs_expr = children.pop().unwrap();
                    if let (Node::Expr(expr1_node), Node::Expr(expr2_node)) = (lhs_expr, rhs_expr) {
                        Expr::BinOp(BinOp::Div, Box::new(expr1_node), Box::new(expr2_node))
                            .evaluate()
                            .map(|expr_node| Node::Expr(expr_node))
                    } else {
                        panic!("Shift/Reduce parsing failure");
                    }
                }
            )
        ),
        // Exponentiation of two expressions
        ProductionRule::new(
            EXPR,
            vec![
                Symbol::Nonterminal(EXPR),
                Symbol::Terminal(Token::Exp),
                Symbol::Nonterminal(EXPR),
            ],
            Tag::new(
                op_disambiguation_map.get(&Token::Exp).unwrap().clone(),
                &move |mut children| {
                    let lhs_expr = children.pop().unwrap();
                    let _exp = children.pop();
                    let rhs_expr = children.pop().unwrap();
                    if let (Node::Expr(expr1_node), Node::Expr(expr2_node)) = (lhs_expr, rhs_expr) {
                        Expr::BinOp(BinOp::Exp, Box::new(expr1_node), Box::new(expr2_node))
                            .evaluate()
                            .map(|expr_node| Node::Expr(expr_node))
                    } else {
                        panic!("Shift/Reduce parsing failure");
                    }
                }
            )
        ),
        // Function invocation
        ProductionRule::new(
            EXPR,
            vec![
                Symbol::Terminal(Token::Id(Default::default())),
                Symbol::Terminal(Token::LParen),
                Symbol::Nonterminal(PARAM),
                Symbol::Terminal(Token::RParen)
            ],
            Tag::new(
                DisambiguationTag::empty(),
                &move |mut children| {
                    let function_name = children.pop().unwrap();
                    let _lparen = children.pop();
                    let params = children.pop().unwrap();
                    let _rparen = children.pop();

                    if let (Node::Terminal(Token::Id(function_name)), Node::Parameter(params)) = (function_name, params) {
                        Expr::Invocation(function_name, params).evaluate().map(|expr_node| Node::Expr(expr_node))
                    } else {
                        panic!("Shift/Reduce parsing failure.");
                    }

                }
            )
        ),
        // ---Parameter Production---
        // Parameter may be empty
        ProductionRule::new(
            PARAM,
            vec![Symbol::Epsilon],
            Tag::new(
                DisambiguationTag::empty(),
                &move |_| {
                    Ok(Node::Parameter(vec![]))
                }
            )
        ),
        // A list of comma separated expressions
        ProductionRule::new(
            PARAM,
            vec![
                Symbol::Nonterminal(EXPR),
                Symbol::Nonterminal(PARAM_R)
            ],
            Tag::new(
                DisambiguationTag::empty(),
                &move |mut children| {
                    let expr = children.pop().unwrap();
                    let param_r = children.pop().unwrap();
                    if let (Node::Expr(expr_node), Node::Remainder(remainder)) = (expr, param_r) {
                        let mut new_param = vec![expr_node];
                        new_param.extend(remainder.into_iter());
                        Ok(Node::Parameter(new_param))
                    } else {
                        panic!("Shift/Reduce parsing failure.");
                    }
                }
            )
        ),
        // ---Parameter Remainder Production---
        // Parameter Remainder may be empty
        ProductionRule::new(
            PARAM_R,
            vec![Symbol::Epsilon],
            Tag::new(
                DisambiguationTag::empty(),
                &move |_| {
                    Ok(Node::Remainder(vec![]))
                }
            )
        ),
        // The "remainder" of a parameter is a comma followed by an expression and another remainder
        ProductionRule::new(
            PARAM_R,
            vec![
                Symbol::Terminal(Token::Comma),
                Symbol::Nonterminal(EXPR),
                Symbol::Nonterminal(PARAM_R)
            ],
            Tag::new(
                DisambiguationTag::empty(),
                &move |mut children| {
                    let _comma = children.pop().unwrap();
                    let expr = children.pop().unwrap();
                    let param_r = children.pop().unwrap();
                    if let (Node::Expr(expr_node), Node::Remainder(remainder)) = (expr, param_r) {
                        let mut new_remainder = vec![expr_node];
                        new_remainder.extend(remainder.into_iter());
                        Ok(Node::Remainder(new_remainder))
                    } else {
                        panic!("Shift/Reduce parsing failure.");
                    }
                }
            )
        ),
    ];
    // Use parser to generate action table
    let lr_automata = LRAutomata::generate(production_rules, 0);
    let (mut action_table, conflicts) = lr_automata.build_action_table();

    // Resolve conflicts
    for conflict in conflicts.iter() {
        let production = &lr_automata.grammar_context.production_rules[conflict.reduce_production_id];
        let conflict_symbol = &conflict.query.1;
        if let Symbol::Terminal(token) = conflict_symbol {
            let production_disambiguation = &production.tag.disambiguation;
            let conflict_disambiguation = op_disambiguation_map.get(token).expect(format!("Cannot resolve conflict: {:?}", conflict).as_str());
            if let Some(production_precedence) = production_disambiguation.precedence {
                if let Some(conflict_precedence) = conflict_disambiguation.precedence {
                    match production_precedence.cmp(&conflict_precedence) {
                        Ordering::Less => {
                            // Shift to keep parsing ... don't use default reduce
                            *action_table.get_mut(&conflict.query).unwrap() = Action::Shift(conflict.shift_target);
                        },
                        Ordering::Equal => {
                            // With equal precedence, check associativity. Left associativity means reduce (default), and
                            // right associativity means shift to keep parsing the right side
                            if let Some(Associativity::Right) = production_disambiguation.associativity {
                                *action_table.get_mut(&conflict.query).unwrap() = Action::Shift(conflict.shift_target)
                            }
                        },
                        Ordering::Greater => {
                            // Use the default reduce behavior
                        },
                    }
                } else {
                    panic!("Cannot resolve conflict: {:?}", conflict);
                }
            } else {
                panic!("Cannot resolve conflict: {:?}", conflict);
            }
        } else {
            panic!("Terminal has to be a conflict symbol.");
        }
    }

    return (lr_automata.grammar_context, action_table);
}

pub fn generate_parse_tree(
    input: &Vec<Token>,
    action_table: &ActionTable,
    grammar_context: &GrammarContext<Nonterminal, Tag>
) -> Result<Node, String>
{
    println!("{:?}", input);
    let mut next_index = 0;

    let mut state_stack = vec![0];
    // node_stack keeps track of node information in the parse tree
    let mut node_stack: Vec<Node> = vec![];
    while next_index <= input.len() {
        let action_query =
            if next_index < input.len() {
                (*state_stack.last().unwrap(), Symbol::Terminal(input[next_index].normalize()))
            } else {
                // We've read the entire input. Insert an end marker symbol.
                (*state_stack.last().unwrap(), Symbol::EndMarker)
            };
        let action = action_table.get(&action_query);
        if action.is_none() {
            return Err(format!("Error at token {}", next_index));
        }
        match action.unwrap() {
            Action::Shift(target_state_id) => {
                // println!("Shifted {:?}", input[next_index]);
                state_stack.push(*target_state_id);
                node_stack.push(Node::Terminal(input[next_index].clone()));
                next_index += 1;
            },
            Action::Reduce(reduce_production_id, effective_rule_size) => {
                let production_rule = grammar_context.production_rules.get(*reduce_production_id).unwrap();
                // println!("Reduced last {} symbol(s) on the stack with production {}: {}", effective_rule_size, production_rule.nonterminal, reduce_production_id);

                let mut sdt_parameters: Vec<Node> = vec![];
                // Order of nodes representing each symbol for the reduce rule is in reverse order.
                for _ in 0..*effective_rule_size {
                    state_stack.pop();
                    sdt_parameters.push(node_stack.pop().unwrap());
                }
                let current_state_id = *state_stack.last().unwrap();
                let nonterminal_shift = action_table.get(&(current_state_id, Symbol::Nonterminal(production_rule.nonterminal))).unwrap();
                if let Action::Shift(s) = nonterminal_shift {
                    state_stack.push(*s);
                    let generated_node = (grammar_context.production_rules[*reduce_production_id].tag.sdt_instructions)(sdt_parameters);
                    match generated_node {
                        Ok(node) => {
                            node_stack.push(node);
                        },
                        Err(msg) => {
                            return Err(format!("Error at token {}: {}", next_index, msg));
                        }
                    }
                } else {
                    panic!("Action table is broken.");
                }
            }
            Action::Accept => {
                // println!("Accept!!!");
                // Current state should be S -> E$. but because we never did a reduce the last state
                // on the node_stack should be E
                return node_stack.pop().ok_or(format!("We messed up."));
            },
        }
    }
    return Err(format!("We really messed up"));
}
