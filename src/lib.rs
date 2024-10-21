// Copyright 2024 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! `SingleVec` is `Vec`-like container type optimized for storing a single item.
//!
//! 0 or 1 items are stored internally as a standard `Option` -
//! which can be kept on the stack -
//! but falls back to a standard `Vec` for multiple items -
//! which are stored on the heap.
//!
//! Although `SingleVec` shares many of the same traits and methods as `Vec`,
//! it also shares many of the same methods as `Option` and `Iterator`
//! where appropriate.
//! Since only a single optional item is intended to be the common case,
//! those methods can avoid iteration altogether.
//!
//! ## Other Features
//! * `serde` provides `Serialize` and `Deserialize` support,
//!   provided that the inner type also has the same implementation.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

#[cfg(feature = "serde")]
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
#[cfg(feature = "serde")]
use serde::ser::{Serialize, SerializeSeq, Serializer};

/// A `Vec`-like type optimized for storing 0 or 1 items.
#[derive(Clone, Debug)]
pub enum SingleVec<T> {
    /// The common case of 0 or 1 items.
    One(Option<T>),
    /// The fallback case of 2 or more items.
    Many(Vec<T>),
}

impl<T> SingleVec<T> {
    /// Constructs a new, empty `SingleVec`
    #[inline]
    pub fn new() -> Self {
        Self::One(None)
    }

    /// Appends an item to the back of the collection.
    ///
    /// # Example
    /// ```
    /// use singlevec::SingleVec;
    /// let mut v = SingleVec::default();
    /// v.push(1);
    /// assert_eq!(v.as_slice(), &[1]);
    /// v.push(2);
    /// assert_eq!(v.as_slice(), &[1, 2]);
    /// ```
    #[inline]
    pub fn push(&mut self, i: T) {
        // hint to the compiler that pushing more than one
        // item into singlevec is intended to be the
        // exceptional case

        #[cold]
        fn move_to_vec<T>(o: &mut Option<T>, i: T) -> Vec<T> {
            let mut v = Vec::with_capacity(2);
            v.extend(o.take());
            v.push(i);
            v
        }

        match self {
            Self::One(o @ None) => {
                *o = Some(i);
            }
            Self::One(o @ Some(_)) => {
                *self = Self::Many(move_to_vec(o, i));
            }
            Self::Many(v) => {
                v.push(i);
            }
        }
    }

    /// Removes the last element from a `SingleVec` and returns it,
    /// or None if it is empty.
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let mut v = SingleVec::from([1, 2, 3]);
    /// assert_eq!(v.pop(), Some(3));
    /// assert_eq!(v.as_slice(), &[1, 2]);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let mut v = SingleVec::from([1]);
    /// assert_eq!(v.pop(), Some(1));
    /// assert_eq!(v.pop(), None);
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        match self {
            Self::One(o) => o.take(),
            Self::Many(v) => v.pop(),
        }
    }

    /// Extracts a slice containing the entire `SingleVec`.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::One(o) => o.as_slice(),
            Self::Many(v) => v.as_slice(),
        }
    }

    /// Extracts a mutable slice containing the entire `SingleVec`.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::One(o) => o.as_mut_slice(),
            Self::Many(v) => v.as_mut_slice(),
        }
    }

    /// Clears the `SingleVec`, removing all values.
    ///
    /// # Example
    /// ```
    /// use singlevec::SingleVec;
    /// let mut v = SingleVec::from([1, 2, 3]);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Uses a closure to determine which elements should remain in the output.
    /// Given an element the closure must return `true` or `false`.
    /// The remaining items are those for which the closure returns `true`.
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from([0i32, 1, 2]);
    /// assert_eq!(v.filter(|x| x.is_positive()).as_slice(), &[1, 2]);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from([0, 1, 2]);
    /// assert_eq!(v.filter(|x| *x > 1).as_slice(), &[2]);
    /// ```
    #[inline]
    pub fn filter(self, f: impl FnMut(&T) -> bool) -> Self {
        match self {
            Self::One(o) => Self::One(o.filter(f)),
            Self::Many(v) => v.into_iter().filter(f).collect(),
        }
    }

    /// Maps a `SingleVec<T>` to a `SingleVec<U>`
    /// by applying a conversion function to each element.
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from(["Hello"]);
    /// assert_eq!(v.map(|s| s.len()), [5].into());
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from(["Hello", "World!"]);
    /// assert_eq!(v.map(|s| s.len()), [5, 6].into());
    /// ```
    #[inline]
    pub fn map<U>(self, f: impl FnMut(T) -> U) -> SingleVec<U> {
        match self {
            Self::One(o) => SingleVec::One(o.map(f)),
            Self::Many(v) => SingleVec::Many(v.into_iter().map(f).collect()),
        }
    }

    /// Combines both a filter and map into a single operation.
    ///
    /// The returned `SingleVec` contains only items for which
    /// the supplied closure returns `Some(value)`.
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from(Some("5"));
    /// assert_eq!(v.filter_map(|s| s.parse().ok()), [5].into());
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from(["1", "two", "NaN", "four", "5"]);
    /// assert_eq!(v.filter_map(|s| s.parse().ok()), [1, 5].into());
    /// ```
    #[inline]
    pub fn filter_map<U>(self, mut f: impl FnMut(T) -> Option<U>) -> SingleVec<U> {
        match self {
            Self::One(None) => SingleVec::One(None),
            Self::One(Some(t)) => SingleVec::One(f(t)),
            Self::Many(v) => v.into_iter().filter_map(f).collect(),
        }
    }

    /// Reduces `SingleVec` to a single item by repeatedly applying a reducing operation.
    ///
    /// The reducing function is a closure with two arguments:
    /// an accumulator, and an element.
    ///
    /// If the `SingleVec` contains a single item, returns that item.
    ///
    /// For a `SingleVec` with more than one item this is the same as fold()
    /// with the first element as the initial accumulator value,
    /// folding every subsequent element into it.
    ///
    /// # Examples
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from([1, 2, 3]);
    /// assert_eq!(v.reduce(|acc, x| acc + x), Some(6));
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from([1]);
    /// assert_eq!(v.reduce(|acc, x| acc + x), Some(1));
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::default();
    /// assert_eq!(v.reduce(|acc: i32, x| acc + x), None);
    /// ```
    #[inline]
    pub fn reduce(self, f: impl FnMut(T, T) -> T) -> Option<T> {
        match self {
            Self::One(o) => o,
            Self::Many(v) => v.into_iter().reduce(f),
        }
    }

    /// Folds every `SingleVec` element into an accumulator by applying a closure.
    ///
    /// Takes an initial value and a closure with two arguments:
    /// an accumulator, and an element.
    /// The closure returns the value that the accumulator should have
    /// for the next iteration.
    ///
    /// The initial value is the value the accumulator will have on the first call.
    ///
    /// Returns the final accumulator
    ///
    /// # Example
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from([1, 2, 3]);
    /// assert_eq!(v.fold(0, |acc, x| acc + x), 6);
    /// ```
    #[inline]
    pub fn fold<B>(self, init: B, mut f: impl FnMut(B, T) -> B) -> B {
        match self {
            Self::One(None) => init,
            Self::One(Some(o)) => f(init, o),
            Self::Many(v) => v.into_iter().fold(init, f),
        }
    }

    /// Zips `SingleVec` with another `SingleVec`
    ///
    /// # Examples
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v1 = SingleVec::from([1]);
    /// let v2 = SingleVec::from([2]);
    /// assert_eq!(v1.zip(v2), [(1, 2)].into());
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v1 = SingleVec::from([1, 2]);
    /// let v2 = SingleVec::from([3, 4]);
    /// assert_eq!(v1.zip(v2), [(1, 3), (2, 4)].into());
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v1 = SingleVec::from([1, 2, 3]);
    /// let v2 = SingleVec::from([4, 5]);
    /// assert_eq!(v1.zip(v2), [(1, 4), (2, 5)].into());
    /// ```
    #[inline]
    pub fn zip<U>(self, other: SingleVec<U>) -> SingleVec<(T, U)> {
        match (self, other) {
            (Self::One(x), SingleVec::One(y)) => SingleVec::One(x.zip(y)),
            (i, j) => i.into_iter().zip(j).collect(),
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Removes all elements e for which `f(&e)` returns `false`.
    /// This method operates in place, visiting each element exactly
    /// once in the original order,
    /// and preserves the order of the retained elements.
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let mut v = SingleVec::from([1, 2, 3, 4]);
    /// v.retain(|&x| x % 2 == 0);
    /// assert_eq!(v.as_slice(), [2, 4]);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let mut v = SingleVec::from([1]);
    /// v.retain(|&x| x % 2 == 0);
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn retain(&mut self, mut f: impl FnMut(&T) -> bool) {
        match self {
            Self::One(o) => {
                if o.as_ref().is_some_and(|x| !f(x)) {
                    *o = None;
                }
            }
            Self::Many(v) => v.retain(f),
        }
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    ///
    /// Removes all elements e such that `f(&mut e)` returns `false`.
    /// This method operates in place, visiting each element exactly
    /// once in the original order,
    /// and preserves the order of the retained elements.
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let mut v = SingleVec::from([1, 2, 3, 4]);
    /// v.retain_mut(|x| if *x <= 3 {
    ///     *x *= 10;
    ///     true
    /// } else {
    ///     false
    /// });
    /// assert_eq!(v.as_slice(), [10, 20, 30]);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let mut v = SingleVec::from([1]);
    /// v.retain_mut(|x| if *x <= 3 {
    ///     *x *= 10;
    ///     true
    /// } else {
    ///     false
    /// });
    /// assert_eq!(v.as_slice(), [10]);
    /// ```
    #[inline]
    pub fn retain_mut(&mut self, mut f: impl FnMut(&mut T) -> bool) {
        match self {
            Self::One(o) => {
                if o.as_mut().is_some_and(|x| !f(x)) {
                    *o = None;
                }
            }
            Self::Many(v) => v.retain_mut(f),
        }
    }
}

impl<T> SingleVec<Option<T>> {
    /// Removes exactly one level of nesting from a `SingleVec`
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from([Some(1), Some(2), None]);
    /// assert_eq!(v.flatten().as_slice(), &[1, 2]);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from([Some(Some(1))]);
    /// assert_eq!(v.flatten().as_slice(), &[Some(1)]);
    /// ```
    #[inline]
    pub fn flatten(self) -> SingleVec<T> {
        match self {
            Self::One(None) => SingleVec::One(None),
            Self::One(Some(inner)) => SingleVec::One(inner),
            Self::Many(v) => v.into_iter().flatten().collect(),
        }
    }

    /// Transposes a `SingleVec` of `Option` into a `Option` of `SingleVec`.
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    ///
    /// let input: SingleVec<Option<i32>> = SingleVec::from([Some(1)]);
    /// let output: Option<SingleVec<i32>> = Some(SingleVec::from([1]));
    /// assert_eq!(input.transpose(), output);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    ///
    /// let input: SingleVec<Option<char>> = SingleVec::from([Some('a'), None]);
    /// assert!(input.transpose().is_none());
    /// ```
    #[inline]
    pub fn transpose(self) -> Option<SingleVec<T>> {
        match self {
            Self::One(None) => None,
            Self::One(Some(inner)) => Some(SingleVec::One(inner)),
            Self::Many(v) => v.into_iter().collect::<Option<_>>().map(SingleVec::Many),
        }
    }
}

impl<T> SingleVec<SingleVec<T>> {
    /// Removes exactly one level of nesting from a `SingleVec`
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let v: SingleVec<SingleVec<i32>> = [[1].into(), [2].into(), [].into()].into();
    /// assert_eq!(v.flatten().as_slice(), &[1, 2]);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v: SingleVec<SingleVec<i32>> = [[1].into()].into();
    /// assert_eq!(v.flatten().as_slice(), &[1]);
    /// ```
    #[inline]
    pub fn flatten(self) -> SingleVec<T> {
        match self {
            Self::One(None) => SingleVec::One(None),
            Self::One(Some(inner)) => inner,
            Self::Many(v) => v.into_iter().flatten().collect(),
        }
    }
}

impl<T> SingleVec<Vec<T>> {
    /// Removes exactly one level of nesting from a `SingleVec`
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let v: SingleVec<Vec<i32>> = [vec![1], vec![2], vec![]].into();
    /// assert_eq!(v.flatten().as_slice(), &[1, 2]);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v: SingleVec<Vec<i32>> = [vec![1, 2, 3]].into();
    /// assert_eq!(v.flatten().as_slice(), &[1, 2, 3]);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v: SingleVec<Vec<i32>> = [vec![1]].into();
    /// assert_eq!(v.flatten().as_slice(), &[1]);
    /// ```
    #[inline]
    pub fn flatten(self) -> SingleVec<T> {
        match self {
            Self::One(None) => SingleVec::One(None),
            Self::One(Some(inner)) => SingleVec::Many(inner),
            Self::Many(v) => v.into_iter().flatten().collect(),
        }
    }
}

impl<T, U> SingleVec<(T, U)> {
    /// Unzips `SingleVec` containing a tuple into a tuple of two `SingleVec`s
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from([(1, 2)]);
    /// assert_eq!(v.unzip(), ([1].into(), [2].into()));
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    /// let v = SingleVec::from([(1, 2), (3, 4)]);
    /// assert_eq!(v.unzip(), ([1, 3].into(), [2, 4].into()));
    /// ```
    #[inline]
    pub fn unzip(self) -> (SingleVec<T>, SingleVec<U>) {
        match self {
            Self::One(None) => (SingleVec::One(None), SingleVec::One(None)),
            Self::One(Some((t, u))) => (SingleVec::One(Some(t)), SingleVec::One(Some(u))),
            Self::Many(v) => v.into_iter().unzip(),
        }
    }
}

impl<T, E> SingleVec<Result<T, E>> {
    /// Transposes a `SingleVec` of `Result` into a `Result` of `SingleVec`,
    /// using the first error encoutered, if any.
    ///
    /// # Examples
    /// ```
    /// use singlevec::SingleVec;
    ///
    /// #[derive(Debug, Eq, PartialEq)]
    /// struct SomeErr;
    ///
    /// let input: SingleVec<Result<i32, SomeErr>> = SingleVec::from([Ok(1)]);
    /// let output: Result<SingleVec<i32>, SomeErr> = Ok(SingleVec::from([1]));
    /// assert_eq!(input.transpose(), output);
    /// ```
    ///
    /// ```
    /// use singlevec::SingleVec;
    ///
    /// #[derive(Debug, Eq, PartialEq)]
    /// struct SomeErr(i32);
    ///
    /// let input: SingleVec<Result<char, SomeErr>> = SingleVec::from([Ok('a'), Err(SomeErr(2)), Err(SomeErr(3))]);
    /// let output: Result<SingleVec<char>, SomeErr> = Err(SomeErr(2));
    /// assert_eq!(input.transpose(), output);
    /// ```
    #[inline]
    pub fn transpose(self) -> Result<SingleVec<T>, E> {
        match self {
            Self::One(o) => o.transpose().map(SingleVec::One),
            Self::Many(v) => v.into_iter().collect::<Result<_, _>>().map(SingleVec::Many),
        }
    }
}

impl<T> Default for SingleVec<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// Although SingleVec::One and SingleVec::Many store data differently,
// they should compare and hash the same if their contents are the same
// without the end user having to know which "mode" the SingleVec
// is in.

impl<T: core::cmp::Eq> core::cmp::Eq for SingleVec<T> {}

impl<T: core::cmp::PartialEq> core::cmp::PartialEq for SingleVec<T> {
    #[inline]
    fn eq(&self, other: &SingleVec<T>) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<T: core::cmp::Ord> core::cmp::Ord for SingleVec<T> {
    #[inline]
    fn cmp(&self, other: &SingleVec<T>) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: core::cmp::PartialOrd> core::cmp::PartialOrd for SingleVec<T> {
    #[inline]
    fn partial_cmp(&self, other: &SingleVec<T>) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: core::hash::Hash> core::hash::Hash for SingleVec<T> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, h: &mut H) {
        self.as_slice().hash(h)
    }
}

impl<T> core::ops::Deref for SingleVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> core::ops::DerefMut for SingleVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Extend<T> for SingleVec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for v in iter {
            self.push(v);
        }
    }
}

impl<T> FromIterator<T> for SingleVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut s = Self::default();
        s.extend(iter);
        s
    }
}

impl<T> IntoIterator for SingleVec<T> {
    type Item = T;

    type IntoIter =
        EitherIterator<<Option<T> as IntoIterator>::IntoIter, <Vec<T> as IntoIterator>::IntoIter>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::One(o) => EitherIterator::I(o.into_iter()),
            Self::Many(v) => EitherIterator::J(v.into_iter()),
        }
    }
}

impl<'t, T> IntoIterator for &'t SingleVec<T> {
    type Item = &'t T;

    type IntoIter = <&'t [T] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<T> From<Option<T>> for SingleVec<T> {
    #[inline]
    fn from(o: Option<T>) -> Self {
        Self::One(o)
    }
}

impl<T> From<Vec<T>> for SingleVec<T> {
    #[inline]
    fn from(mut v: Vec<T>) -> Self {
        match v.len() {
            0 | 1 => Self::One(v.pop()),
            _ => Self::Many(v),
        }
    }
}

impl<T> From<SingleVec<T>> for Vec<T> {
    #[inline]
    fn from(v: SingleVec<T>) -> Self {
        match v {
            SingleVec::One(None) => Vec::new(),
            SingleVec::One(Some(x)) => vec![x],
            SingleVec::Many(v) => v,
        }
    }
}

impl<T> TryFrom<SingleVec<T>> for Option<T> {
    type Error = Vec<T>;

    #[inline]
    fn try_from(v: SingleVec<T>) -> Result<Self, Self::Error> {
        match v {
            SingleVec::One(o) => Ok(o),
            SingleVec::Many(mut v) if v.len() < 2 => Ok(v.pop()),
            SingleVec::Many(v) => Err(v),
        }
    }
}

impl<T, const N: usize> From<[T; N]> for SingleVec<T> {
    #[inline]
    fn from(v: [T; N]) -> Self {
        v.into_iter().collect()
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for SingleVec<T> {
    #[must_use]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for element in self.iter() {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for SingleVec<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(SingleVecVisitor(core::marker::PhantomData))
    }
}

#[cfg(feature = "serde")]
struct SingleVecVisitor<T>(core::marker::PhantomData<T>);

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>> Visitor<'de> for SingleVecVisitor<T> {
    type Value = SingleVec<T>;

    fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut v = SingleVec::default();

        while let Some(value) = seq.next_element()? {
            v.push(value);
        }

        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::SingleVec;

    #[test]
    fn iter_test() {
        let v = SingleVec::from([1, 2, 3]);
        assert_eq!(v.iter().copied().sum::<i32>(), 6);

        let v = SingleVec::from([1, 2, 3]);
        assert_eq!((&v).into_iter().copied().sum::<i32>(), 6);
    }

    #[test]
    fn index_test() {
        let mut v = SingleVec::from([1, 2, 3]);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);

        v[0] = 3;
        v[1] = 2;
        v[2] = 1;

        assert_eq!(v[0], 3);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 1);
    }

    #[test]
    fn eq_test() {
        let v1: SingleVec<i32> = SingleVec::One(None);
        let v2 = SingleVec::Many(vec![]);
        assert_eq!(v1, v2);

        let v1: SingleVec<i32> = SingleVec::One(Some(1));
        let v2 = SingleVec::Many(vec![1]);
        assert_eq!(v1, v2);
    }

    #[test]
    fn ord_test() {
        let v1: SingleVec<i32> = SingleVec::One(Some(1));
        let v2 = SingleVec::Many(vec![2]);
        assert!(v1 < v2);
    }

    #[test]
    fn hash_test() {
        let mut h = std::collections::HashSet::new();
        h.insert(SingleVec::One(Some(1)));
        assert!(h.contains(&SingleVec::Many(vec![1])));
    }
}

/// An iterator which can be one of two possible variants
/// but iterate over the same type.
///
/// Checks the iterator variant on each iteration,
/// but the expectation is that the iterator length
/// will usually be very short.
#[derive(Clone, Debug)]
pub enum EitherIterator<I, J> {
    /// The first iterator variant
    I(I),
    /// The second iterator variant
    J(J),
}

impl<I: Iterator, J: Iterator<Item = I::Item>> Iterator for EitherIterator<I, J> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::I(i) => i.next(),
            Self::J(j) => j.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::I(i) => i.size_hint(),
            Self::J(j) => j.size_hint(),
        }
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        match self {
            Self::I(i) => i.count(),
            Self::J(j) => j.count(),
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        match self {
            Self::I(i) => i.last(),
            Self::J(j) => j.last(),
        }
    }

    #[inline]
    fn for_each<F>(self, f: F)
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        match self {
            Self::I(i) => i.for_each(f),
            Self::J(j) => j.for_each(f),
        }
    }

    #[inline]
    fn collect<B>(self) -> B
    where
        B: FromIterator<Self::Item>,
        Self: Sized,
    {
        match self {
            Self::I(i) => i.collect(),
            Self::J(j) => j.collect(),
        }
    }

    #[inline]
    fn partition<B, F>(self, f: F) -> (B, B)
    where
        Self: Sized,
        B: Default + Extend<Self::Item>,
        F: FnMut(&Self::Item) -> bool,
    {
        match self {
            Self::I(i) => i.partition(f),
            Self::J(j) => j.partition(f),
        }
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        match self {
            Self::I(i) => i.fold(init, f),
            Self::J(j) => j.fold(init, f),
        }
    }

    #[inline]
    fn reduce<F>(self, f: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(Self::Item, Self::Item) -> Self::Item,
    {
        match self {
            Self::I(i) => i.reduce(f),
            Self::J(j) => j.reduce(f),
        }
    }

    #[inline]
    fn all<F>(&mut self, f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        match self {
            Self::I(i) => i.all(f),
            Self::J(j) => j.all(f),
        }
    }

    #[inline]
    fn any<F>(&mut self, f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        match self {
            Self::I(i) => i.any(f),
            Self::J(j) => j.any(f),
        }
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        match self {
            Self::I(i) => i.find(predicate),
            Self::J(j) => j.find(predicate),
        }
    }

    #[inline]
    fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Option<B>,
    {
        match self {
            Self::I(i) => i.find_map(f),
            Self::J(j) => j.find_map(f),
        }
    }

    #[inline]
    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        Self: Sized,
        P: FnMut(Self::Item) -> bool,
    {
        match self {
            Self::I(i) => i.position(predicate),
            Self::J(j) => j.position(predicate),
        }
    }

    #[inline]
    fn max(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        match self {
            Self::I(i) => i.max(),
            Self::J(j) => j.max(),
        }
    }

    #[inline]
    fn min(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        match self {
            Self::I(i) => i.min(),
            Self::J(j) => j.min(),
        }
    }

    #[inline]
    fn max_by_key<B, F>(self, f: F) -> Option<Self::Item>
    where
        B: Ord,
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
    {
        match self {
            Self::I(i) => i.max_by_key(f),
            Self::J(j) => j.max_by_key(f),
        }
    }

    #[inline]
    fn max_by<F>(self, compare: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> core::cmp::Ordering,
    {
        match self {
            Self::I(i) => i.max_by(compare),
            Self::J(j) => j.max_by(compare),
        }
    }

    #[inline]
    fn min_by_key<B, F>(self, f: F) -> Option<Self::Item>
    where
        B: Ord,
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
    {
        match self {
            Self::I(i) => i.min_by_key(f),
            Self::J(j) => j.min_by_key(f),
        }
    }

    #[inline]
    fn min_by<F>(self, compare: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> core::cmp::Ordering,
    {
        match self {
            Self::I(i) => i.min_by(compare),
            Self::J(j) => j.min_by(compare),
        }
    }

    #[inline]
    fn sum<S>(self) -> S
    where
        Self: Sized,
        S: core::iter::Sum<Self::Item>,
    {
        match self {
            Self::I(i) => i.sum(),
            Self::J(j) => j.sum(),
        }
    }

    #[inline]
    fn product<P>(self) -> P
    where
        Self: Sized,
        P: core::iter::Product<Self::Item>,
    {
        match self {
            Self::I(i) => i.product(),
            Self::J(j) => j.product(),
        }
    }

    #[inline]
    fn cmp<K>(self, other: K) -> core::cmp::Ordering
    where
        K: IntoIterator<Item = Self::Item>,
        Self::Item: core::cmp::Ord,
        Self: Sized,
    {
        match self {
            Self::I(i) => i.cmp(other),
            Self::J(j) => j.cmp(other),
        }
    }

    #[inline]
    fn partial_cmp<K>(self, other: K) -> Option<core::cmp::Ordering>
    where
        K: IntoIterator,
        Self::Item: PartialOrd<<K as IntoIterator>::Item>,
        Self: Sized,
    {
        match self {
            Self::I(i) => i.partial_cmp(other),
            Self::J(j) => j.partial_cmp(other),
        }
    }

    #[inline]
    fn eq<K>(self, other: K) -> bool
    where
        K: IntoIterator,
        Self::Item: PartialEq<<K as IntoIterator>::Item>,
        Self: Sized,
    {
        match self {
            Self::I(i) => i.eq(other),
            Self::J(j) => j.eq(other),
        }
    }

    #[inline]
    fn ne<K>(self, other: K) -> bool
    where
        K: IntoIterator,
        Self::Item: PartialEq<<K as IntoIterator>::Item>,
        Self: Sized,
    {
        match self {
            Self::I(i) => i.ne(other),
            Self::J(j) => j.ne(other),
        }
    }

    #[inline]
    fn lt<K>(self, other: K) -> bool
    where
        K: IntoIterator,
        Self::Item: PartialOrd<<K as IntoIterator>::Item>,
        Self: Sized,
    {
        match self {
            Self::I(i) => i.lt(other),
            Self::J(j) => j.lt(other),
        }
    }

    #[inline]
    fn le<K>(self, other: K) -> bool
    where
        K: IntoIterator,
        Self::Item: PartialOrd<<K as IntoIterator>::Item>,
        Self: Sized,
    {
        match self {
            Self::I(i) => i.le(other),
            Self::J(j) => j.le(other),
        }
    }

    #[inline]
    fn gt<K>(self, other: K) -> bool
    where
        K: IntoIterator,
        Self::Item: PartialOrd<<K as IntoIterator>::Item>,
        Self: Sized,
    {
        match self {
            Self::I(i) => i.gt(other),
            Self::J(j) => j.gt(other),
        }
    }

    #[inline]
    fn ge<K>(self, other: K) -> bool
    where
        K: IntoIterator,
        Self::Item: PartialOrd<<K as IntoIterator>::Item>,
        Self: Sized,
    {
        match self {
            Self::I(i) => i.ge(other),
            Self::J(j) => j.ge(other),
        }
    }
}

impl<I: DoubleEndedIterator, J: DoubleEndedIterator<Item = I::Item>> DoubleEndedIterator
    for EitherIterator<I, J>
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::I(i) => i.next_back(),
            Self::J(j) => j.next_back(),
        }
    }
}

impl<I: ExactSizeIterator, J: ExactSizeIterator<Item = I::Item>> ExactSizeIterator
    for EitherIterator<I, J>
{
}

impl<I: core::iter::FusedIterator, J: core::iter::FusedIterator<Item = I::Item>>
    core::iter::FusedIterator for EitherIterator<I, J>
{
}
