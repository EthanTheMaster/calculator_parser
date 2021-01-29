use crate::parser::{ProductionRule, Symbol, LRAutomata, Action, ActionQuery, GrammarContext};
use lexer::{Token};
use std::collections::HashMap;
use std::iter::FromIterator;
use std::cmp::Ordering;
use crate::calculator::ast::{Node, Expr, UnaryOp, BinOp};

pub mod ast;
pub mod lexer;

pub type Nonterminal = &'static str;

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

pub type ActionTable = HashMap<ActionQuery<Nonterminal>, Action>;
pub fn generate_action_table<'a>() ->  (GrammarContext<Nonterminal, Tag<'a>>, ActionTable) {
    let op_disambiguation_map: HashMap<Token, DisambiguationTag> = HashMap::from_iter(vec![
        (Token::Add, DisambiguationTag::new(Some(Associativity::Left), Some(0))),
        (Token::Sub, DisambiguationTag::new(Some(Associativity::Left), Some(0))),
        (Token::Mult, DisambiguationTag::new(Some(Associativity::Left), Some(1))),
        (Token::Div, DisambiguationTag::new(Some(Associativity::Left), Some(1))),
        (Token::Exp, DisambiguationTag::new(Some(Associativity::Right), Some(2))),

        // Treat these next tokens as if initiating multiplication
        (Token::Int(Default::default()).normalize(), DisambiguationTag::new(Some(Associativity::Left), Some(1))),
        (Token::Float(Default::default()).normalize(), DisambiguationTag::new(Some(Associativity::Left), Some(1))),
        (Token::Id(Default::default()).normalize(), DisambiguationTag::new(Some(Associativity::Left), Some(1))),
        (Token::LParen, DisambiguationTag::new(Some(Associativity::Left), Some(1))),

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
                    if let Node::Terminal(token) = int_terminal {
                        Expr::evaluate_token(&token).map(|expr_node| Node::Expr(expr_node))
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
                     if let Node::Terminal(token) = float_terminal {
                         Expr::evaluate_token(&token).map(|expr_node| Node::Expr(expr_node))
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
        // Treat side by side expressions as multiplication.
        ProductionRule::new(
            EXPR,
            vec![
                Symbol::Nonterminal(EXPR),
                Symbol::Nonterminal(EXPR)
            ],
            Tag::new(
                op_disambiguation_map.get(&Token::Mult).unwrap().clone(),
                &move |mut children| {
                    let expr1 = children.pop().unwrap();
                    let expr2 = children.pop().unwrap();
                    if let (Node::Expr(expr_node1), Node::Expr(expr_node2)) = (expr1, expr2) {
                        Expr::BinOp(BinOp::Mult, Box::new(expr_node1), Box::new(expr_node2))
                            .evaluate()
                            .map(|expr_node| Node::Expr(expr_node))
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
    let action_table = lr_automata.build_action_table();
    let mut conflict_free_action_table = HashMap::new();
    // Resolve conflicts
    for (query, mut action_list) in action_table {
        // Conflict occurs when there are multiple actions that can be done for a given query
        if action_list.len() > 1 {
            // Iterate through entire list and pick best by comparing the current item with the
            // known best, similar to finding the "max" in a list.
            let mut chosen_action = action_list.pop().unwrap();
            while !action_list.is_empty() {
                let current = action_list.pop().unwrap();
                // Given chosen_action and current, construct a pair where the first component
                // is a shift (if possible). This is prevent redundant code.
                let (action1, action2) = if let Action::Shift(_) = chosen_action {
                    (&chosen_action, &current)
                } else {
                    (&current, &chosen_action)
                };
                match (action1, action2) {
                    // Consider shift/reduce conflict
                    (Action::Shift(target_state), Action::Reduce(rule_id, effective_size)) => {
                        let lookahead_symbol = if let Symbol::Terminal(lookahead) = &query.1 {
                            lookahead
                        } else {
                            panic!("Lookahead needs to be a terminal.");
                        };
                        // Look at precedence tag
                        let rule_disambiguation = &lr_automata.grammar_context.production_rules[*rule_id].tag.disambiguation;
                        let symbol_disambiguation = op_disambiguation_map
                            .get(lookahead_symbol)
                            .expect("Attempted to disambiguate symbol which has no data.");
                        match (rule_disambiguation.precedence, symbol_disambiguation.precedence) {
                            (Some(p1), Some(p2)) => {
                                let shift = Action::Shift(*target_state);
                                let reduce = Action::Reduce(*rule_id, *effective_size);
                                match p1.cmp(&p2) {
                                    Ordering::Less => {
                                        // reduce action has lower precedence than lookahead so shift to not
                                        // prematurely reduce
                                        chosen_action = shift;
                                    },
                                    Ordering::Equal => {
                                        // With equal precedence, look at reduce operation's associativity
                                        match rule_disambiguation.associativity {
                                            Some(Associativity::Left) => {
                                                chosen_action = reduce;
                                            },
                                            Some(Associativity::Right) => {
                                                chosen_action = shift;
                                            },
                                            None => {
                                                panic!("There is not enough data to resolve shift/reduce conflict.");
                                            },
                                        }
                                    },
                                    Ordering::Greater => {
                                        // reduce action has higher precedence so reduce
                                        chosen_action = reduce;
                                    },
                                }
                            },
                            _ => {
                                panic!("There is not enough data to resolve shift/reduce conflict.");
                            }
                        }

                    },
                    // Consider reduce/reduce conflict
                    (Action::Reduce(rule_id1, effective_size1), Action::Reduce(rule_id2, effective_size2)) => {
                        println!("WARNING: Reduce/Reduce conflict between {} and {} given query {:?}.", rule_id1, rule_id2, query);
                        // Resolve reduce/reduce conflict by choosing the rule that is listed "higher" or "first" when
                        // listing production rules from top to bottom (ie the smaller the index the higher priority
                        // or precedence it has).
                        if rule_id1 < rule_id2 {
                            chosen_action = Action::Reduce(*rule_id1, *effective_size1)
                        } else {
                            chosen_action = Action::Reduce(*rule_id2, *effective_size2)
                        }
                    },
                    _ => {
                        panic!("Impossible situation");
                    }
                }
            }
            // chosen_action is now the action with highest precedence
            conflict_free_action_table.insert(query, chosen_action);
        } else {
            // There is no conflict. Take the only item in the action list
            conflict_free_action_table.insert(query, action_list.pop().unwrap());
        }
    }

    return (lr_automata.grammar_context, conflict_free_action_table);
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
                println!("Shifted {:?}", input[next_index]);
                state_stack.push(*target_state_id);
                node_stack.push(Node::Terminal(input[next_index].clone()));
                next_index += 1;
            },
            Action::Reduce(reduce_production_id, effective_rule_size) => {
                let production_rule = grammar_context.production_rules.get(*reduce_production_id).unwrap();
                println!("Reduced last {} symbol(s) on the stack with production {}: {}", effective_rule_size, production_rule.nonterminal, reduce_production_id);

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
                println!("Accept!!!");
                // Current state should be S -> E$. but because we never did a reduce the last state
                // on the node_stack should be E
                return node_stack.pop().ok_or(format!("We messed up."));
            },
        }
    }
    return Err(format!("We really messed up"));
}
