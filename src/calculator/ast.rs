use crate::calculator::lexer::Token;

// Holds type data and corresponding value
#[derive(Debug, Clone)]
pub enum Values {
    Float(f64),
    Int(i64)
}

impl Values {
    fn coerce(value: &Values, target_type: &Values) -> Option<Values> {
        match (value, target_type) {
            (Values::Int(n), Values::Int(_)) => {Some(Values::Int(*n))},
            (Values::Float(n), Values::Float(_)) => {Some(Values::Float(*n))},
            (Values::Int(n), Values::Float(_)) => {Some(Values::Float(*n as f64))},
            _ => {None}
        }
    }
}

#[derive(Debug)]
pub enum Node {
    Terminal(Token),
    Expr(ExprNode),
    Parameter(Vec<ExprNode>),
    Remainder(Vec<ExprNode>)
}

#[derive(Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mult,
    Div,
    Exp
}

#[derive(Debug)]
pub enum UnaryOp {
    Neg,
}

#[derive(Debug)]
pub enum Expr {
    Number(Values),
    BinOp(BinOp, Box<ExprNode>, Box<ExprNode>),
    UnaryOp(UnaryOp, Box<ExprNode>),
    Invocation(String, Vec<ExprNode>),
}

impl Expr {
    pub fn evaluate(self) -> Result<ExprNode, String> {
        match self {
            Expr::Number(value) => {
                Ok(ExprNode {
                    value: value.clone(),
                    expr_variant: Expr::Number(value)
                })
            },
            Expr::BinOp(op, lhs, rhs) => {
                let value = match op {
                    BinOp::Add => {
                        match (&lhs.value, &rhs.value) {
                            (Values::Int(n1), Values::Int(n2)) => {Some(Values::Int(n1+n2))},
                            (Values::Float(n1), Values::Float(n2)) => {Some(Values::Float(n1+n2))},
                            (Values::Int(n1), Values::Float(n2)) => {Some(Values::Float(*n1 as f64 + n2))},
                            (Values::Float(n1), Values::Int(n2)) => {Some(Values::Float(n1 + *n2 as f64))},
                            // _ => {None}
                        }
                    },
                    BinOp::Sub => {
                        match (&lhs.value, &rhs.value) {
                            (Values::Int(n1), Values::Int(n2)) => {Some(Values::Int(n1-n2))},
                            (Values::Float(n1), Values::Float(n2)) => {Some(Values::Float(n1-n2))},
                            (Values::Int(n1), Values::Float(n2)) => {Some(Values::Float(*n1 as f64 - n2))},
                            (Values::Float(n1), Values::Int(n2)) => {Some(Values::Float(n1 - *n2 as f64))},
                            // _ => {None}
                        }
                    },
                    BinOp::Mult => {
                        match (&lhs.value, &rhs.value) {
                            (Values::Int(n1), Values::Int(n2)) => {Some(Values::Int(n1*n2))},
                            (Values::Float(n1), Values::Float(n2)) => {Some(Values::Float(n1*n2))},
                            (Values::Int(n1), Values::Float(n2)) => {Some(Values::Float(*n1 as f64 * n2))},
                            (Values::Float(n1), Values::Int(n2)) => {Some(Values::Float(n1 * *n2 as f64))},
                            // _ => {None}
                        }
                    },
                    BinOp::Div => {
                        match (&lhs.value, &rhs.value) {
                            (Values::Int(n1), Values::Int(n2)) => {Some(Values::Int(n1/n2))},
                            (Values::Float(n1), Values::Float(n2)) => {Some(Values::Float(n1/n2))},
                            (Values::Int(n1), Values::Float(n2)) => {Some(Values::Float(*n1 as f64 / n2))},
                            (Values::Float(n1), Values::Int(n2)) => {Some(Values::Float(n1 / *n2 as f64))},
                            // _ => {None}
                        }
                    },
                    BinOp::Exp => {
                        match (&lhs.value, &rhs.value) {
                            (Values::Int(n1), Values::Int(n2)) => {Some(Values::Int(n1.pow(*n2 as u32)))},
                            (Values::Float(n1), Values::Float(n2)) => {Some(Values::Float(n1.powf(*n2)))},
                            (Values::Int(n1), Values::Float(n2)) => {Some(Values::Float((*n1 as f64).powf(*n2)))},
                            (Values::Float(n1), Values::Int(n2)) => {Some(Values::Float(n1.powi(*n2 as i32)))},
                            // _ => {None}
                        }
                    },
                };
                match value {
                    None => {
                        Err(format!("Type Error"))
                    },
                    Some(v) => {
                        Ok(ExprNode {
                            value: v,
                            expr_variant: Expr::BinOp(op, lhs, rhs)
                        })
                    },
                }
            },
            Expr::UnaryOp(op, x) => {
                match op {
                    UnaryOp::Neg => {
                        let value = match x.value {
                            Values::Float(n) => {Values::Float(-n)},
                            Values::Int(n) => {Values::Int(-n)},
                        };
                        Ok(ExprNode {
                            value,
                            expr_variant: Expr::UnaryOp(op, x)
                        })
                    },
                }
            },
            Expr::Invocation(function, params) => {
                // Compute function result into Values enum
                let value = match function.as_str() {
                    "cos" => {
                        // Check number of parameters
                        if params.len() == 1 {
                            // Coerce parameter into float
                            match Values::coerce(&params[0].value, &Values::Float(0.0)) {
                                Some(Values::Float(n)) => {
                                    Ok(Values::Float(n.cos()))
                                },
                                _ => {
                                    Err(format!("Type error."))
                                },
                            }
                        } else {
                            Err(format!("Expected 1 argument. Found {}.", params.len()))
                        }
                    },
                    "sin" => {
                        // Check number of parameters
                        if params.len() == 1 {
                            // Coerce parameter into float
                            match Values::coerce(&params[0].value, &Values::Float(0.0)) {
                                Some(Values::Float(n)) => {
                                    Ok(Values::Float(n.sin()))
                                },
                                _ => {
                                    Err(format!("Type error."))
                                },
                            }
                        } else {
                            Err(format!("Expected 1 argument. Found {}.", params.len()))
                        }
                    },
                    "tan" => {
                        // Check number of parameters
                        if params.len() == 1 {
                            // Coerce parameter into float
                            match Values::coerce(&params[0].value, &Values::Float(0.0)) {
                                Some(Values::Float(n)) => {
                                    Ok(Values::Float(n.tan()))
                                },
                                _ => {
                                    Err(format!("Type error."))
                                },
                            }
                        } else {
                            Err(format!("Expected 1 argument. Found {}.", params.len()))
                        }
                    },
                    "exp" => {
                        // Check number of parameters
                        if params.len() == 1 {
                            // Coerce parameter into float
                            match Values::coerce(&params[0].value, &Values::Float(0.0)) {
                                Some(Values::Float(n)) => {
                                    Ok(Values::Float(n.exp()))
                                },
                                _ => {
                                    Err(format!("Type error."))
                                },
                            }
                        } else {
                            Err(format!("Expected 1 argument. Found {}.", params.len()))
                        }
                    },
                    "ln" => {
                        // Check number of parameters
                        if params.len() == 1 {
                            // Coerce parameter into float
                            match Values::coerce(&params[0].value, &Values::Float(0.0)) {
                                Some(Values::Float(n)) => {
                                    Ok(Values::Float(n.ln()))
                                },
                                _ => {
                                    Err(format!("Type error."))
                                },
                            }
                        } else {
                            Err(format!("Expected 1 argument. Found {}.", params.len()))
                        }
                    },
                    "abs" => {
                        // Check number of parameters
                        if params.len() == 1 {
                            // Coerce parameter into float
                            match Values::coerce(&params[0].value, &Values::Float(0.0)) {
                                Some(Values::Float(n)) => {
                                    Ok(Values::Float(n.abs()))
                                },
                                _ => {
                                    Err(format!("Type error."))
                                },
                            }
                        } else {
                            Err(format!("Expected 1 argument. Found {}.", params.len()))
                        }
                    },
                    "sqrt" => {
                        // Check number of parameters
                        if params.len() == 1 {
                            // Coerce parameter into float
                            match Values::coerce(&params[0].value, &Values::Float(0.0)) {
                                Some(Values::Float(n)) => {
                                    Ok(Values::Float(n.sqrt()))
                                },
                                _ => {
                                    Err(format!("Type error."))
                                },
                            }
                        } else {
                            Err(format!("Expected 1 argument. Found {}.", params.len()))
                        }
                    },
                    "float" => {
                        // Check number of parameters
                        if params.len() == 1 {
                            // Coerce parameter into float
                            match Values::coerce(&params[0].value, &Values::Float(0.0)) {
                                Some(Values::Float(n)) => {
                                    Ok(Values::Float(n))
                                },
                                _ => {
                                    Err(format!("Type error."))
                                },
                            }
                        } else {
                            Err(format!("Expected 1 argument. Found {}.", params.len()))
                        }
                    },
                    "int" => {
                        // Check number of parameters
                        if params.len() == 1 {
                            // Coerce parameter into float
                            match &params[0].value {
                                Values::Int(n) => {
                                    Ok(Values::Int(*n))
                                },
                                Values::Float(n) => {
                                    Ok(Values::Int(*n as i64))
                                }
                            }
                        } else {
                            Err(format!("Expected 1 argument. Found {}.", params.len()))
                        }
                    },
                    "log" => {
                        // Check number of parameters
                        if params.len() == 2 {
                            // Coerce parameter into float
                            let base = Values::coerce(&params[0].value, &Values::Float(0.0));
                            let param = Values::coerce(&params[1].value, &Values::Float(0.0));
                            match (base, param) {
                                (Some(Values::Float(base)), Some(Values::Float(param))) => {
                                    Ok(Values::Float(param.log(base)))
                                },
                                _ => {
                                    Err(format!("Type Error."))
                                }
                            }
                        } else {
                            Err(format!("Expected 2 arguments. Found {}.", params.len()))
                        }
                    }
                    _ => Err(format!("Unknown function: `{}`", function))
                };
                // Construct ExprNode
                match value {
                    Ok(value) => {
                        Ok(ExprNode {
                            value,
                            expr_variant: Expr::Invocation(function, params)
                        })
                    },
                    Err(msg) => {
                        Err(msg)
                    },
                }

            },
        }
    }
}

#[derive(Debug)]
pub struct ExprNode {
    pub value: Values,
    pub expr_variant: Expr,
}