use std::collections::HashMap;
use std::hash::Hash;

#[cfg(feature = "enum-map")]
use enum_map::{EnumArray, EnumMap};
#[cfg(feature = "enum-map")]
use std::ops::IndexMut;

pub trait MapLike<K, V> {
    fn new_() -> Self;
    fn get_mut(&mut self, k: &K) -> Option<&mut V>;
    fn insert_(&mut self, k: K, v: V) -> Option<V>;
}

// TODO: I don't understand why I can't re-use the HashMap function names
impl<K, V> MapLike<K, V> for HashMap<K, V>
where
    K: Eq + Hash,
{
    fn new_() -> Self {
        HashMap::new()
    }
    fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        HashMap::get_mut(self, k)
    }
    fn insert_(&mut self, k: K, v: V) -> Option<V> {
        self.insert(k, v)
    }
}

// TODO: I don't understand the EnumArray trait and why it needs the value
// TODO: I don't understand why index_mut doesn't take a reference
#[cfg(feature = "enum-map")]
impl<K, V> MapLike<K, Vec<V>> for EnumMap<K, Vec<V>>
where
    K: Eq + EnumArray<Vec<V>> + Clone,
{
    fn new_() -> Self {
        Self::default()
    }
    fn get_mut(&mut self, k: &K) -> Option<&mut Vec<V>> {
        Some(self.index_mut(k.clone()))
    }
    fn insert_(&mut self, k: K, mut v: Vec<V>) -> Option<Vec<V>> {
        std::mem::swap(self.index_mut(k.clone()), &mut v);
        Some(v)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[cfg(feature = "enum-map")]
    use enum_map::{enum_map, Enum};

    // TODO: would be nice if Enum could use Result and Option
    #[derive(Debug, Eq, PartialEq, Hash, Clone)]
    #[cfg_attr(feature = "enum-map", derive(Enum))]
    enum Flup {
        A,
        B,
        C,
    }

    struct Nothing {}

    fn blub<F, T, X>(f: F, _x: X)
    where
        F: Fn() -> T,
        X: EnumArray<T>
    {
        let m = enum_map! { Flup::A => f(), Flup::B => f(), _ => f()};
    }

    fn bla() {
        blub::<_, Nothing, _>(|| Nothing {}, Flup::A);
    }

    #[cfg(feature = "enum-map")]
    impl Flup {
        fn from_char(c: &char) -> Self {
            match c {
                'a' => Flup::A,
                'b' => Flup::B,
                _ => Flup::C,
            }
        }
    }

    #[cfg(feature = "enum-map")]
    #[test]
    fn t() {
        let mut my_map = enum_map! { Flup::A => vec!['a'],
        Flup::B => vec!['b'],
        Flup::C => vec!['c']};

        let rv = my_map.insert_(Flup::A, vec!['A']);
        assert_eq!(rv, vec!['a'].into());
        assert_eq!(my_map[Flup::A], vec!['A']);
    }

    #[test]
    fn test_hash_map() {
        let v = vec![Flup::A, Flup::B, Flup::A, Flup::A, Flup::B, Flup::A];
        let rv = group_into_hash_map(v.into_iter(), |f| f.clone());
        let mut cmp = HashMap::new();
        cmp.insert(Flup::A, vec![Flup::A, Flup::A, Flup::A, Flup::A]);
        cmp.insert(Flup::B, vec![Flup::B, Flup::B]);
        assert_eq!(cmp, rv);
    }

    #[cfg(feature = "enum-map")]
    #[test]
    fn test_enum_map_kv() {
        let v = vec!['a', 'b', 'c', 'a', 'e'];
        let rv = group_into::<_, _, _, _, EnumMap<_, _>>(v.into_iter(), Flup::from_char);
        let cmp = enum_map! {
        Flup::A => vec!['a', 'a'],
        Flup::B => vec!['b'],
        _ => vec!['c', 'e']};
        assert_eq!(cmp, rv);
    }

    #[cfg(feature = "enum-map")]
    #[test]
    fn test_enum_map() {
        let v = vec![Flup::A, Flup::B, Flup::A, Flup::A, Flup::B, Flup::A];
        let rv = group_into::<_, _, _, _, EnumMap<_, _>>(v.into_iter(), |f| f.clone());
        let cmp = enum_map! {
        Flup::A => vec![Flup::A, Flup::A, Flup::A, Flup::A],
        Flup::B => vec![Flup::B, Flup::B],
        _ => vec![]};
        assert_eq!(cmp, rv);
    }
}

pub fn group_into_hash_map<K, V, F, I>(iter: I, f: F) -> HashMap<K, Vec<V>>
where
    I: Iterator<Item = V>,
    F: Fn(&V) -> K,
    K: Eq + Hash + std::fmt::Debug,
    K: Eq + Hash,
    V: std::fmt::Debug,
{
    group_into::<K, V, F, I, HashMap<K, Vec<V>>>(iter, f)
}

// TODO: instead of Vec<V>, it should just be anything that has push(&mut self, v) or similar like
// fold.
pub fn group_into<K, V, F, I, M>(mut iter: I, f: F) -> M
where
    I: Iterator<Item = V>,
    F: Fn(&V) -> K,
    K: Eq + std::fmt::Debug,
    M: MapLike<K, Vec<V>>,
    V: std::fmt::Debug,
{
    let mut retval = M::new_();
    while let Some(v) = iter.next() {
        let k = f(&v);
        match retval.get_mut(&k) {
            Some(t) => t.push(v),
            None => {
                retval.insert_(k, vec![v]);
            }
        };
    }
    retval
}
