/*
This implementation is based on Sections 4.6 - 4.8 of Compilers Principles, Techniques, & Tools
Second Edition. The parser is based on the LR(1) parser implementation in Section 4.7.
*/

use crate::calculator::lexer::Token;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::fmt::{Debug, Formatter};
use std::fmt;
use std::hash::Hash;

// Set up a Context-Free Grammar
#[derive(Hash, Eq, Clone, Debug)]
// Id is used to identify nonterminals
pub enum Symbol<Id>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    Nonterminal(Id),
    Terminal(Token),
    Epsilon,
    // This symbol represents the end of the input/end of start production
    EndMarker,
}

impl<Id> PartialEq for Symbol<Id>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    // Check if two symbols are equal where terminal equality does not regard attribute information
    fn eq(&self, other: &Self) -> bool {
        return match self {
            Symbol::Nonterminal(nonterminal_id) => {
                if let Symbol::Nonterminal(other_id) = other {
                    *nonterminal_id == *other_id
                } else {
                    false
                }
            },
            Symbol::Terminal(token) => {
                if let Symbol::Terminal(other_token) = other {
                    token.normalize() == other_token.normalize()
                } else {
                    false
                }
            },
            Symbol::Epsilon => {
                if let Symbol::Epsilon = other {
                    true
                } else {
                    false
                }
            },
            Symbol::EndMarker => {
                if let Symbol::EndMarker = other {
                    true
                } else {
                    false
                }
            },
        }
    }
}

// Tag type represents any sort of accompanying information that might be useful to keep about the
// production during parsing, particularly when resolving parsing conflicts.
pub struct ProductionRule<Id, Tag>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    pub nonterminal: Id,
    // String of symbols that represents one possible production for the nonterminal
    // associated with id.
    pub rule: Vec<Symbol<Id>>,
    pub tag: Tag,
}

impl<Id, Tag> ProductionRule<Id, Tag>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    pub fn new(id: Id, rule: Vec<Symbol<Id>>, tag: Tag) -> Self {
        ProductionRule {
            nonterminal: id,
            rule,
            tag
        }
    }
}


type FirstTable<Id> = HashMap<Id, HashSet<Symbol<Id>>>;
// Given a list of production rules, compute first which returns a mapping from
// nonterminal ids to a list of symbols(all except nonterminals)
pub fn first<Id, Tag>(productions: &Vec<ProductionRule<Id, Tag>>) -> FirstTable<Id>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    let mut has_table_converged = false;
    let mut res: FirstTable<Id> = HashMap::new();
    // Initialize table
    for production in productions {
        res.insert(production.nonterminal.clone(), HashSet::new());
    }

    while !has_table_converged {
        has_table_converged = true; // We will attempt to falsify this
        for production in productions {
            let tentative_first = first_symbol_string(&production.rule, &res);
            for first in tentative_first {
                // if we successfully added a new symbol into the table, we have not converged onto
                // the final table solution keep iterating
                if res.get_mut(&production.nonterminal).unwrap().insert(first) {
                    has_table_converged = false;
                }
            }
        }
    }

    return res;
}

// Given a list of grammar symbols, compute first on it making use of a supplied pre-computed
// FirstTable.
fn first_symbol_string<Id>(symbol_string: &Vec<Symbol<Id>>, context: &FirstTable<Id>) -> HashSet<Symbol<Id>>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    let mut res = HashSet::new();
    // a symbol string is nullable iff it can derive the empty string/epsilon ... we will attempt
    // to falsify this in the loop
    for symbol in symbol_string {
        let is_nullable: bool;
        match symbol {
            Symbol::Nonterminal(nonterminal_id) => {
                // Import symbols in the first of this nonterminal into our result
                for first_symbol in context.get(&nonterminal_id).unwrap() {
                    match first_symbol {
                        Symbol::Nonterminal(_) => {
                            panic!("Nonterminal cannot be in first!");
                        },
                        Symbol::Terminal(token) => {
                            res.insert(Symbol::Terminal(token.normalize()));
                        },
                        Symbol::Epsilon => {
                            // Do nothing
                        },
                        Symbol::EndMarker => {
                            res.insert(Symbol::EndMarker);
                        },
                    }
                }
                // symbol string is nullable up to this point if current terminal is nullable
                is_nullable = context.get(nonterminal_id).unwrap().contains(&Symbol::Epsilon);
            }
            Symbol::Terminal(token) => {
                res.insert(Symbol::Terminal(token.normalize()));
                // terminals/tokens are not nullable
                is_nullable = false;
            },
            Symbol::Epsilon => {
                // epsilon symbol is trivially nullable
                is_nullable = true;
            },
            Symbol::EndMarker => {
                res.insert(Symbol::EndMarker);
                // end marker cannot be nullable
                is_nullable = false;
            },
        }
        // There is no point in looking at next symbols as we just met a symbol than cannot be
        // nullable and first of next symbols are not relevant.
        if !is_nullable {
            return res;
        }
    }

    // We reach this point iff we've iterated through the entire string and each symbol
    // was nullable. Therefore the entire symbol string is nullable, so add epsilon to first to
    // signal this fact.
    res.insert(Symbol::Epsilon);

    return res;
}

// Creates a map that maps nonterminal ids to their production rules
pub fn build_nonterminal_production_map<Id, Tag>(production_rules: &Vec<ProductionRule<Id, Tag>>) -> HashMap<Id, Vec<usize>>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    let mut res = HashMap::new();
    for (production_id, production) in production_rules.iter().enumerate() {
        let rules_option = res.get_mut(&production.nonterminal);
        match rules_option {
            None => {
                res.insert(production.nonterminal.clone(), vec![production_id]);
            },
            Some(list) => {
                list.push(production_id);
            },
        }
    }
    return res;
}

#[derive(Eq, PartialEq, Hash, Debug, Clone)]
pub struct LRItem {
    // Id that identifies a particular production rule. Id will be derived from the index of the
    // production rule in a Vec
    rule_id: usize,
    // Dot position represents where the dot is located in a LR item E -> α.β where α and β
    // are non-epsilon grammar symbols. The symbol following the dot (.) is the next symbol to be
    // matched
    dot_position: usize
}

impl LRItem {
    pub fn new(rule_id: usize, dot_position: usize) -> Self {
        LRItem {
            rule_id,
            dot_position
        }
    }
}

// Structure that holds precomputed information about the grammar that is useful when generating
// a parser
pub struct GrammarContext<Id, Tag>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    pub production_rules: Vec<ProductionRule<Id, Tag>>,
    // Maps nonterminal ids to a list of indices that refer to production rules in `production_rules`
    nonterminal_production_map: HashMap<Id, Vec<usize>>,
    first_table: FirstTable<Id>
}

impl<Id, Tag> GrammarContext<Id, Tag>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    fn new(production_rules: Vec<ProductionRule<Id, Tag>>) -> GrammarContext<Id, Tag> {
        let nonterminal_production_map = build_nonterminal_production_map(&production_rules);
        let first_table = first(&production_rules);
        GrammarContext {
            production_rules,
            nonterminal_production_map,
            first_table
        }
    }
}


#[derive(Eq, Clone)]
pub struct LRState<Id>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    // Functions as a set of LR(0) items with each item associated with a set
    // of lookahead symbols for LR(1) parsing
    items: HashMap<LRItem, HashSet<Symbol<Id>>>,
    // usize should refer to the global id of an LRState
    transitions: HashMap<Symbol<Id>, usize>,
}

impl<Id: PartialEq + Eq + Clone + Hash + Debug> Debug for LRState<Id> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LRState")
            .field("items", &self.items)
            .field("transitions", &self.transitions)
            .finish()
    }
}

impl<Id> PartialEq for LRState<Id>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    fn eq(&self, other: &Self) -> bool {
        return self.items == other.items;
    }
}

impl<Id> LRState<Id>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    pub fn new(items: Vec<(LRItem, HashSet<Symbol<Id>>)>) -> Self {
        let res = LRState {
            items: HashMap::from_iter(items.into_iter()),
            transitions: Default::default(),
        };
        return res;
    }

    fn is_item_in_set(&self, item: &LRItem, lookahead: &Symbol<Id>) -> bool {
        return if self.items.contains_key(&item) {
            if let Some(lookaheads) = self.items.get(&item) {
                lookaheads.contains(lookahead)
            } else {
                false
            }
        } else {
            false
        }
    }

    // Applies the Closure operation on an LR State
    // - production_rules essentially maps a production rule id to its rule
    // - terminal_production_map maps a terminal id to a list of production rule ids
    pub fn closure<Tag>(&mut self, grammar_context: &GrammarContext<Id, Tag>) {
        let mut has_converged = false;
        while !has_converged {
            has_converged = true; // We will attempt to falsify this
            // List of LRItem-Lookahead symbol pairs to add that are generated as we look at each
            // LR(1) item
            let mut derived_items: Vec<(LRItem, Symbol<Id>)> = vec![];
            // items that have epsilon as next symbol are not needed
            let mut items_to_delete: Vec<LRItem> = vec![];
            for (item, lookaheads) in self.items.iter() {
                let production = grammar_context.production_rules.get(item.rule_id).unwrap();
                let next_symbol = production.rule.get(item.dot_position);
                if let Some(symbol) = next_symbol {
                    match symbol {
                        Symbol::Nonterminal(nonterminal_id) => {
                            // Get productions associated with this nonterminal and add them as items
                            let nonterminal_productions = grammar_context.nonterminal_production_map.get(nonterminal_id).unwrap();
                            for rule_id in nonterminal_productions {
                                let derived_item = LRItem { rule_id: *rule_id, dot_position: 0 };

                                // Figure out the lookahead symbol(s) for this derived item
                                let lookahead_string = &production.rule[item.dot_position+1..];
                                // In order to determine the lookaheads of the derived item, consider we are looking at the item
                                // (A -> α.Bβ, x) with lookahead symbol x. The symbols in first(βx) will be part of B's lookahead.
                                // We need to iterate through all the current rule's lookaheads to determine derived lookaheads.
                                for lookahead in lookaheads {
                                    // Determine first(βx)
                                    let mut lookahead_string_vec = lookahead_string.to_vec();
                                    lookahead_string_vec.push(lookahead.clone());
                                    let derived_lookaheads = first_symbol_string(&lookahead_string_vec, &grammar_context.first_table);
                                    for derived_lookahead in derived_lookaheads.iter() {
                                        match derived_lookahead {
                                            Symbol::Nonterminal(_) => {panic!("Nonterminal cannot be a first symbol.")},
                                            Symbol::Terminal(token) => {
                                                if !self.is_item_in_set(&derived_item, &Symbol::Terminal(token.normalize())) {
                                                    has_converged = false;
                                                    derived_items.push((derived_item.clone(), Symbol::Terminal(token.normalize())));
                                                }
                                            },
                                            Symbol::Epsilon => {
                                                // Do not consider epsilon as a lookahead
                                            },
                                            Symbol::EndMarker => {
                                                if !self.is_item_in_set(&derived_item, &Symbol::EndMarker) {
                                                    has_converged = false;
                                                    derived_items.push((derived_item.clone(), Symbol::EndMarker));
                                                }
                                            },
                                        }
                                    }
                                }
                            }
                        }
                        Symbol::Epsilon => {
                            items_to_delete.push(item.clone());
                            // If the next symbol is epsilon, push dot over this symbol and add to
                            // set of items. Lookahead symbols will be the same
                            let derived_item = LRItem {rule_id: item.rule_id, dot_position: item.dot_position + 1};
                            for lookahead in lookaheads.iter() {
                                if !self.is_item_in_set(&derived_item, &lookahead) {
                                    has_converged = false;
                                    derived_items.push((derived_item.clone(), lookahead.clone()))
                                }
                            }
                        },
                        Symbol::Terminal(_) | Symbol::EndMarker => {
                            // Do nothing. There are no productions associated with these symbols
                        },
                    }
                }
            }
            // Import list of derived items with lookaheads into our set of LR(1) items
            for (derived_item, derived_lookahead) in derived_items.into_iter() {
                match self.items.get_mut(&derived_item) {
                    None => {
                        self.items.insert(derived_item, [derived_lookahead].iter().cloned().collect());
                    },
                    Some(lookaheads) => {
                        lookaheads.insert(derived_lookahead);
                    },
                }
            }
            // Delete extraneous items
            for item in items_to_delete {
                self.items.remove(&item);
            }
        }
    }

    // production_rules maps id/indices to production rule
    pub fn valid_next_symbols<Tag>(&self, grammar_context: &GrammarContext<Id, Tag>) -> HashSet<Symbol<Id>> {
        return self.items.iter()
            .map(|(item, _)| {
                let next_symbol_option = grammar_context.production_rules.get(item.rule_id).unwrap().rule.get(item.dot_position);
                match next_symbol_option {
                    // We will later filter out/trash epsilon as a valid next symbol
                    None => {Symbol::Epsilon},
                    Some(next_symbol) => {
                        match next_symbol {
                            Symbol::Nonterminal(nonterminal_id) => {Symbol::Nonterminal(nonterminal_id.clone())}
                            Symbol::Terminal(token) => {Symbol::Terminal(token.normalize())},
                            Symbol::Epsilon => {Symbol::Epsilon},
                            Symbol::EndMarker => {Symbol::EndMarker},
                        }
                    },
                }
            })
            .filter(|symbol| {
                match symbol {
                    Symbol::Epsilon => false,
                    _ => true,
                }
            })
            .collect();
    }

    pub fn goto<Tag>(&self, symbol: &Symbol<Id>, grammar_context: &GrammarContext<Id, Tag>) -> LRState<Id> {
        let mut new_items: HashMap<LRItem, HashSet<Symbol<Id>>> = HashMap::new();
        for (item, lookaheads) in self.items.iter() {
            let rule = grammar_context.production_rules.get(item.rule_id).unwrap();
            let next_symbol_option = rule.rule.get(item.dot_position);
            if let Some(next_symbol) = next_symbol_option {
                if next_symbol == symbol {
                    new_items.insert(LRItem::new(item.rule_id, item.dot_position + 1), lookaheads.clone());
                }
            }
        }

        let mut res = LRState {
            items: new_items,
            transitions: Default::default(),
        };
        res.closure(grammar_context);

        return res;

    }
}

pub struct LRAutomata<Id, Tag>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    // 0th state is the start state
    pub states: Vec<LRState<Id>>,
    pub grammar_context: GrammarContext<Id, Tag>,
    pub start_production_id: usize,
}

// A query consists of the current state id and the next symbol to be consumed in the input
pub type ActionQuery<Id> = (usize, Symbol<Id>);
#[derive(Debug)]
pub enum Action {
    // usize refers to which state to transition to
    Shift(usize),
    // first usize refers to which production rule is used to reduce by and second usize
    // refers to how many symbols to pop off the stack
    Reduce(usize, usize),
    Accept,
}

#[derive(Debug)]
pub struct ShiftReduceConflict<Id>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    pub query: ActionQuery<Id>,
    pub shift_target: usize,
    pub reduce_production_id: usize
}

impl<Id, Tag> LRAutomata<Id, Tag>
    where Id: PartialEq + Eq + Clone + Hash + Debug
{
    fn compute_all_states(mut start_state: LRState<Id>, grammar_context: &GrammarContext<Id, Tag>) -> Vec<LRState<Id>> {
        start_state.closure(&grammar_context);

        // To determine all states in the automata, perform DPS
        let mut explored = vec![];
        let mut unexplored = vec![start_state];

        while !unexplored.is_empty() {
            let current = unexplored.pop().unwrap();
            if !explored.contains(&current) {
                let valid_transition_symbols = current.valid_next_symbols(&grammar_context);
                for symbol in valid_transition_symbols {
                    unexplored.push(current.goto(&symbol, &grammar_context));
                }

                explored.push(current);
            }
        }

        return explored;
    }

    fn compute_transitions(states: &mut Vec<LRState<Id>>, grammar_context: &GrammarContext<Id, Tag>) {
        let states_clone = states.clone();
        for state in states.iter_mut() {
            let valid_transition_symbols = state.valid_next_symbols(grammar_context);
            for symbol in valid_transition_symbols {
                let target = state.goto(&symbol, grammar_context);
                // Search for the target in states ... we have to use `states`'s clone as we are currently
                // iterating through it with a mutable reference into each item
                let id = states_clone.iter()
                    .enumerate().find(|(_, s)| target == **s)
                    .expect("states does not have all states in the automata. Call `compute_all_states`")
                    .0; // Pull out the index component
                if state.transitions.insert(symbol, id).is_some() {
                    panic!("Cannot have two transitions for same symbol.")
                }
            }
        }
    }

    pub fn generate(
        production_rules: Vec<ProductionRule<Id, Tag>>,
        start_production_id: usize
    ) -> LRAutomata<Id, Tag> {
        let grammar_context = GrammarContext::new(production_rules);
        // println!("{:#?}", grammar_context.first_table);
        // println!("{:#?}", grammar_context.nonterminal_production_map);

        let start_state = LRState::new(
            vec![(LRItem::new(start_production_id, 0), [Symbol::EndMarker].iter().cloned().collect())],
        );

        let mut states = LRAutomata::compute_all_states(start_state, &grammar_context);
        LRAutomata::compute_transitions(&mut states, &grammar_context);
        // println!("{:#?}", states[0]);

        let res = LRAutomata {
            states,
            grammar_context,
            start_production_id
        };

        return res;
    }

    // Builds a table that maps action query to an action and allows the parser information to be
    // precomputed for future parsing. The second component of the result has a list of queries
    // that create a shift/reduce conflict. In the event of a conflict, the table will default
    // to a reduce.
    pub fn build_action_table(&self) -> (HashMap<ActionQuery<Id>, Action>, Vec<ShiftReduceConflict<Id>>) {
        let mut res = HashMap::new();
        let mut conflicts = vec![];
        for (state_id, state) in self.states.iter().enumerate() {
            // Compute Shift/Goto Rules
            for (symbol, target_state_id) in state.transitions.iter() {
                match symbol {
                    Symbol::Nonterminal(_) => {
                        res.insert((state_id, symbol.clone()), Action::Shift(*target_state_id));
                    },
                    Symbol::Terminal(token) => {
                        res.insert((state_id, Symbol::Terminal(token.normalize())), Action::Shift(*target_state_id));
                    },
                    Symbol::Epsilon => {panic!("Epsilon should not be a transition symbol.")},
                    Symbol::EndMarker => {
                        // Check if the state has the start production and has the end marker as a valid transition symbol
                        let can_accept = state.items.iter().any(|(item,_)| {
                            item.rule_id == self.start_production_id
                        }) && state.transitions.contains_key(&Symbol::EndMarker);

                        if can_accept {
                            res.insert((state_id, Symbol::EndMarker), Action::Accept);
                        }
                    },
                }
            }

            // Compute Reduce Rules
            for (item, lookaheads) in state.items.iter() {
                // Make sure dot is at the end of the rule which means we matched all previous symbols
                let production_rule = &self.grammar_context.production_rules.get(item.rule_id).unwrap().rule;
                // Reduce rule applies to items where the dot is at the end which represents all symbols have been matched
                if item.dot_position == production_rule.len() {
                    // Compute the number of symbols to pop off the stack
                    let mut effective_rule_size = 0;
                    for sym in production_rule.iter() {
                        if let Symbol::Epsilon = sym {
                            // Epsilon symbol contribute nothing to the length of the rule
                        } else {
                            effective_rule_size += 1;
                        }
                    }

                    // `lookaheads` contains all valid symbols that can follow the current rule
                    for lookahead in lookaheads.iter() {
                        if let Some(Action::Shift(shift_target)) = res.insert((state_id, lookahead.clone()), Action::Reduce(item.rule_id, effective_rule_size)) {
                            // Insertion led to an overwrite which means there was already an action in place. This action
                            // must be a shift from before. Overwrite action with a reduce and note the conflict.
                            let conflict = ShiftReduceConflict {
                                query: (state_id, lookahead.clone()),
                                shift_target,
                                reduce_production_id: item.rule_id
                            };
                            conflicts.push(conflict);
                        }
                    }
                }
            }
        }
        return (res, conflicts);
    }
}
