#![allow(non_snake_case)]
#![feature(allocator_api, alloc_layout_extra)]
//! Soa2, Soa3, ..SoaN are generic collections with an API similar to that of a Vec of tuples but which store
//! the data laid out as a separate slice per field. The advantage of this layout is that when
//! iterating over the data only a subset need be loaded from RAM.
//!
//! This approach is common to game engines, and entity component systems in particular but is
//! applicable anywhere that cache coherency and memory bandwidth are important for performance.
//!
//!
//! # Example
//! ```
//! # use soa_vec::Soa3::Soa3;
//! /// Some 'entity' data.
//! # #[derive(Copy, Clone)]
//! struct Position { x: f64, y: f64 }
//! # #[derive(Copy, Clone)]
//! struct Velocity { dx: f64, dy: f64 }
//! struct ColdData { /* Potentially many fields omitted here */ }
//!
//! # use std::ops::Add;
//! # impl Add<Velocity> for Position { type Output=Self; fn add(self, other: Velocity) -> Self { Self { x: self.x + other.dx, y: self.y + other.dy } } }
//! // Create a vec of entities
//! let mut entities = Soa3::new();
//! entities.push((Position {x: 1.0, y: 2.0}, Velocity { dx: 0.0, dy: 0.5 }, ColdData {}));
//! entities.push((Position {x: 0.0, y: 2.0}, Velocity { dx: 0.5, dy: 0.5 }, ColdData {}));
//!
//! // Update entities. This loop only loads position and velocity data, while skipping over
//! // the ColdData which is not necessary for the physics simulation.
//! let (positions, velocities, _cold) = entities.iters_mut();
//! for (position, velocity) in positions.zip(velocities) {
//! 	*position = *position + *velocity;
//! }
//!
//! // Remove an entity
//! entities.swap_remove(0);
//!
//! // Sort entities by position on y axis
//! // The fields are passed by reference, so velocity and cold data are not loaded
//! // until such time as the items are being swapped which runs in O(N)
//! # use std::cmp;
//! entities.sort_unstable_by(
//! 	|(lh_pos, _, _), (rh_pos, _, _)| lh_pos.y.partial_cmp(&rh_pos.y).unwrap()
//! );
//!
//! // See individual structs for more methods.
//!
//! ```
//!
//!
//! # Nightly
//! This crate has strict requirements for allocations and memory layout and therefore requires the following nightly features:
//! * allocator_api
//! * alloc_layout_extra
//!
//! # Links:
//! * [Github source](https://github.com/That3Percent/soa-vec)
//! * [Crate](https://crates.io/crates/soa-vec)
//!
//! # License
//! [MIT](https://github.com/That3Percent/soa-vec/blob/master/LICENSE)
//!
//! # Related
//! If you like this, you may like these other crates by Zac Burns (That3Percent)
//! * [second-stack](https://github.com/That3Percent/second-stack) A Rust memory allocator for large slices that don't escape the stack.
//! * [js-intern](https://github.com/That3Percent/js-intern) Stores one copy of each distinct JavaScript primitive
//! * [js-object](https://github.com/That3Percent/js-object) A macro for creating JavaScript objects


// TODO: ZSTs break iterators and allocations. https://doc.rust-lang.org/nomicon/vec-zsts.html

/// This macro defines a struct-of-arrays style struct.
/// It need not be called often, just once per count of generic parameters.
macro_rules! soa {

	($name:ident, $t1:ident, $($ts:ident),+) => {
		pub mod $name {
			use second_stack::*;
			use std::{alloc::*, cmp::*, marker::*, ptr::*, slice::*};
			use std::ops::RangeBounds;
			use core::ops::Bound::{Included, Excluded, Unbounded};

			/// Struct of arrays storage with vec API. See module docs for more information.
			pub struct $name<$t1: Sized $(, $ts: Sized)*> {
				len: usize,
				capacity: usize,
				$t1: NonNull<$t1>,
				$($ts: NonNull<$ts>,)*
				_marker: (PhantomData<$t1> $(, PhantomData<$ts>)*),
			}

			impl<$t1: Sized $(, $ts: Sized)*> $name<$t1 $(, $ts)*> {
				/// Creates a new Soa with a capacity of 0
				pub fn new() -> $name<$t1 $(, $ts)*> {
					$name {
						len: 0,
						capacity: 0,
						$t1: NonNull::dangling(),
						$($ts: NonNull::dangling(),)*
						_marker: (PhantomData $(, PhantomData::<$ts>)*),
					}
				}

				/// Constructs a new, empty Soa with the specified capacity.
				/// The soa will be able to hold exactly capacity elements without reallocating. If capacity is 0, the soa will not allocate.
				/// It is important to note that although the returned soa has the capacity specified, the soa will have a zero length.
				pub fn with_capacity(capacity: usize) -> $name<$t1 $(, $ts)*> {
					if capacity == 0 {
						Self::new()
					} else {
						let ($t1 $(,$ts)*) = Self::alloc(capacity);

						Self {
							capacity,
							len: 0,
							$t1: $t1,
							$($ts: $ts,)*
							_marker: (PhantomData $(, PhantomData::<$ts>)*),
						}
					}
				}

				fn dealloc(&mut self) {
					if self.capacity > 0 {
						let layout = Self::layout_for_capacity(self.capacity).layout;
						unsafe { Global.dealloc(self.$t1.cast::<u8>(), layout) }
					}
				}

				/// Allocates and partitions a new region of uninitialized memory
				fn alloc(capacity: usize) -> (NonNull<$t1> $(, NonNull<$ts>)*) {
					unsafe {
						let layouts = Self::layout_for_capacity(capacity);
						let bytes = Global.alloc(layouts.layout).unwrap();
						(
							bytes.cast::<$t1>()
							$(, NonNull::new_unchecked(bytes.as_ptr().add(layouts.$ts) as *mut $ts))*
						)
					}
				}

				fn check_grow(&mut self) {
					unsafe {
						if self.len == self.capacity {
							let capacity = (self.capacity * 2).max(4);
							let ($t1 $(, $ts)*) = Self::alloc(capacity);

							copy_nonoverlapping(self.$t1.as_ptr(), $t1.as_ptr(), self.len);
							$(
								copy_nonoverlapping(self.$ts.as_ptr(), $ts.as_ptr(), self.len);
							)*

							self.dealloc();

							// Assign
							self.$t1 = $t1;
							$(self.$ts = $ts;)*
							self.capacity = capacity;

						}
					}
				}

				/// Returns the number of tuples in the soa, also referred to as its 'length'.
				#[inline(always)]
				pub fn len(&self) -> usize { self.len }

				/// Returns the number of elements the soa can hold without reallocating.
				#[inline(always)]
				pub fn capacity(&self) -> usize { self.capacity }

				/// Returns true if the soa has a length of 0.
				#[inline(always)]
				pub fn is_empty(&self) -> bool { self.len == 0 }

				/// Clears the soa, removing all values.
				/// Note that this method has no effect on the allocated capacity of the soa.
				pub fn clear(&mut self) {
					while self.len > 0 {
						self.pop();
					}
				}

				/// Appends a tuple to the back of a soa.
				pub fn push(&mut self, value: ($t1 $(, $ts)*)) {
					unsafe {
						self.check_grow();
						let ($t1 $(, $ts)*) = value;
						write(self.$t1.as_ptr().add(self.len), $t1);
						$(write(self.$ts.as_ptr().add(self.len), $ts);)*
						self.len += 1;
					}
				}

				/// Inserts an element at position index within each array, shifting all elements after it to the right.
				///
				/// # Panics
				/// Must panic if index > len.
				pub fn insert(&mut self, index: usize, value: ($t1 $(, $ts)*)) {
					let len = self.len;
					assert!(index <= len);
					unsafe {
						self.check_grow(); // TODO: (Performance) In the case where we do grow, this can result in redundant copying.

						let ($t1 $(, $ts)*) = value;

						{
							let p = self.$t1.as_ptr().add(index);
							copy(p, p.offset(1), len - index);
							write(p, $t1);
						}

						$({
							let p = self.$ts.as_ptr().add(index);
							copy(p, p.offset(1), len - index);
							write(p, $ts);
						})*
						self.len = len + 1;
					}
				}

				/// Removes the last tuple from a soa and returns it, or None if it is empty.
				pub fn pop(&mut self) -> Option<($t1 $(, $ts)*)> {
					if self.len == 0 {
						None
					} else {
						self.len -= 1;
						unsafe {
							Some((
								read(self.$t1.as_ptr().add(self.len))
								$(, read(self.$ts.as_ptr().add(self.len)))*
							))
						}
					}
				}

				/// Removes and returns the element at position index within the vector, shifting all elements after it to the left.
				/// # Panics
				/// Must panic if index is out of bounds.
				pub fn remove(&mut self, index: usize) -> ($t1 $(, $ts)*) {
					let len = self.len;
					assert!(index < len);
					unsafe {
						let $t1;
						$(let $ts;)*

						{
							let ptr = self.$t1.as_ptr().add(index);
							$t1 = read(ptr);
							copy(ptr.offset(1), ptr, len - index - 1);
						}

						$({
							let ptr = self.$ts.as_ptr().add(index);
							$ts = read(ptr);
							copy(ptr.offset(1), ptr, len - index - 1);
						})*

						self.len = len - 1;

						($t1 $(, $ts)*)
					}
				}

				/// Removes a tuple from the soa and returns it.
				/// The removed tuple is replaced by the last tuple of the soa.
				/// This does not preserve ordering, but is O(1).
				///
				/// # Panics:
				///  * Must panic if index is out of bounds
				pub fn swap_remove(&mut self, index: usize) -> ($t1 $(, $ts)*) {
					if index >= self.len {
						panic!("Index out of bounds");
					}

					unsafe {
						let $t1 = self.$t1.as_ptr().add(index);
						$(let $ts = self.$ts.as_ptr().add(index);)*

						let v = (
							read($t1)
							$(, read($ts))*
						);

						self.len -= 1;

						if self.len != index {
							copy_nonoverlapping(self.$t1.as_ptr().add(self.len), $t1, 1);
							$(copy_nonoverlapping(self.$ts.as_ptr().add(self.len), $ts, 1);)*
						}

						v
					}
				}

				fn layout_for_capacity(capacity: usize) -> OwnLayout {
					let layout = Layout::array::<$t1>(capacity).unwrap();

					$(let (layout, $ts) = layout.extend(Layout::array::<$ts>(capacity).unwrap()).unwrap();)*

					OwnLayout {
						layout
						$(, $ts)*
					}
				}

				/// Returns a tuple of all the destructured tuples added to this soa.
				#[inline(always)] // Inline for dead code elimination
				pub fn slices<'a>(&self) -> (&'a [$t1] $(, &'a [$ts])*) {
					unsafe {
						(
							from_raw_parts::<'a>(self.$t1.as_ptr(), self.len),
							$(from_raw_parts::<'a>(self.$ts.as_ptr(), self.len),)*
						)
					}
				}

				/// Returns a tuple of iterators over each field in the soa.
				#[inline(always)] // Inline for dead code elimination
				pub fn iters<'a>(&self) -> (Iter<'a, $t1> $(, Iter<'a, $ts>)*) {
					unsafe {
						(
							from_raw_parts::<'a>(self.$t1.as_ptr(), self.len).iter()
							$(, from_raw_parts::<'a>(self.$ts.as_ptr(), self.len).iter())*
						)
					}
				}

				/// Returns a tuple of mutable iterators over each field in the soa.
				#[inline(always)] // Inline for dead code elimination
				pub fn iters_mut<'a>(&mut self) -> (IterMut<'a, $t1> $(, IterMut<'a, $ts>)*) {
					unsafe {
						(
							from_raw_parts_mut::<'a>(self.$t1.as_ptr(), self.len).iter_mut()
							$(, from_raw_parts_mut::<'a>(self.$ts.as_ptr(), self.len).iter_mut())*
						)
					}
				}

				/// Returns a tuple of all the destructured mutable tuples added to this soa.
				#[inline(always)] // Inline for dead code elimination
				pub fn slices_mut<'a>(&self) -> (&'a mut [$t1] $(, &'a mut [$ts])*) {
					unsafe {
						(
							from_raw_parts_mut::<'a>(self.$t1.as_ptr(), self.len),
							$(from_raw_parts_mut::<'a>(self.$ts.as_ptr(), self.len),)*
						)
					}
				}

				/// This is analogous to the index operator in vec, but returns a tuple of references.
				/// ## Panics
				/// * If index is >= len
				pub fn index<'a>(&self, index: usize) -> (&'a $t1 $(, &'a $ts)*) {
					unsafe {
						if index >= self.len {
							panic!("Index out of range");
						}

						(
							&*self.$t1.as_ptr().add(index)
							$(, &*self.$ts.as_ptr().add(index))*
						)
					}
				}

				/// Sorts the soa keeping related data together.
				pub fn sort_unstable_by<F: FnMut((&$t1 $(, &$ts)*), (&$t1 $(, &$ts)*))->Ordering>(&mut self, mut f: F) {
					if self.len < 2 {
						return;
					}
					let mut indices = acquire(0..self.len);

					indices.sort_unstable_by(|a, b| unsafe {
						f(
							(&*self.$t1.as_ptr().add(*a) $(, &*self.$ts.as_ptr().add(*a))*, ),
							(&*self.$t1.as_ptr().add(*b) $(, &*self.$ts.as_ptr().add(*b))*, ),
						)});

					// Example
					// c b d e a
					// 4 1 0 2 3 // indices
					// 2 1 3 4 0 // lookup

					let mut lookup = unsafe { acquire_uninitialized(self.len) };
					for (i, index) in indices.iter().enumerate() {
						lookup[*index] = i;
					}

					let ($t1 $(, $ts)*) = self.slices_mut();

					for i in 0..indices.len() {
						let dest = indices[i]; // The index that should go here
						if i != dest {
							// Swap
							$t1.swap(i, dest);
							$($ts.swap(i, dest);)*

							// Account for swaps that already happened
							indices[lookup[i]] = dest;
							lookup[dest] = lookup[i];
						}
					}
				}

				pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> Drain<$t1 $(, $ts)*> {

					let len = self.len();
					let start = match range.start_bound() {
						Included(&n) => n,
						Excluded(&n) => n + 1,
						Unbounded    => 0,
					};
					let end = match range.end_bound() {
						Included(&n) => n + 1,
						Excluded(&n) => n,
						Unbounded    => len,
					};
					assert!(start <= end);
					assert!(end <= len);

					let iter = unsafe { RawValIter::new(&self, start, end - start) };
					self.len = start;
					Drain {
						tail_start: end,
						tail_len: len - end,
						iter,
						soa: NonNull::from(self),
					}
				}
			}

			struct OwnLayout {
				layout: Layout,
				$($ts: usize,)*
			}

			struct RawValIter<'a, $t1: 'a $(, $ts: 'a)*> {
				$t1: StartEnd<'a, $t1>,
				$($ts: StartEnd<'a, $ts>,)*
			}

			impl<'a, $t1: 'a $(, $ts: 'a)*> RawValIter<'a, $t1 $(, $ts)*> {
				unsafe fn new(soa: &$name<$t1 $(, $ts)*>, start: usize, len: usize) -> Self {
					Self {
						$t1: StartEnd::new(soa.$t1, start, len),
						$($ts: StartEnd::new(soa.$ts, start, len),)*
					}
				}
			}

			impl<'a, $t1: 'a $(, $ts: 'a)*> Iterator for RawValIter<'a, $t1 $(, $ts)*> {
				type Item = ($t1 $(, $ts)*);

				fn next(&mut self) -> Option<Self::Item> {
					if self.$t1.is_empty() {
						None
					} else {
						unsafe { Some((self.$t1.next() $(, self.$ts.next())*)) }
					}
				}
				fn size_hint(&self) -> (usize, Option<usize>) {
					let len = self.$t1.len();
					(len, Some(len))
				}
			}

			impl<'a, $t1 $(, $ts)*> ExactSizeIterator for RawValIter<'a, $t1 $(, $ts)*> {
				fn len(&self) -> usize {
					self.$t1.len()
				}
			}

			impl<'a, $t1: 'a $(, $ts: 'a)*> DoubleEndedIterator for RawValIter<'a, $t1 $(, $ts)*> {
				fn next_back(&mut self) -> Option<Self::Item> {
					if self.$t1.is_empty() {
						None
					} else {
						unsafe { Some((self.$t1.next_back() $(, self.$ts.next_back())*)) }
					}
				}
			}

			// TODO: Consider using Slice::Iter
			struct StartEnd<'a, T: 'a> {
				start: *const T,
				end: *const T, // TODO: Note the comment about ZST here: https://doc.rust-lang.org/src/core/slice/mod.rs.html#3285-3291 in Iter<'a, T: 'a>
				_marker: PhantomData<&'a T>,
			}

			impl<'a, T: 'a> StartEnd<'a, T> {
				pub unsafe fn new(ptr: NonNull<T>, offset: usize, len: usize) -> Self {
					let start = if len == 0 { ptr.as_ptr() } else { ptr.as_ptr().add(offset) };
					Self {
						start,
						end: if len == 0 { start } else { start.add(len) },
						_marker: PhantomData,
					}
				}

				pub unsafe fn next(&mut self) -> T {
					let result = read(self.start);
					self.start = self.start.offset(1);
					result
				}

				pub unsafe fn next_back(&mut self) -> T {
					self.end = self.end.offset(-1);
					read(self.end)
				}

				pub fn is_empty(&self) -> bool {
					self.start == self.end
				}

				pub fn len(&self) -> usize {
					(self.end as usize - self.start as usize) / std::mem::size_of::<T>()
				}
			}

			// TODO: These may be multiple lifetimes, with the first lifetime bound by the others. Consider (though proc_macro is needed to create lifetime names here)
			pub struct Drain<'a, $t1: 'a $(, $ts: 'a)*> {
				soa: NonNull<$name<$t1 $(, $ts)*>>,
				iter: RawValIter<'a, $t1 $(, $ts)*>,
				tail_start: usize,
				tail_len: usize,
			}

			impl<'a, $t1: 'a $(, $ts: 'a)*> Iterator for Drain<'a, $t1 $(, $ts)*> {
				type Item = ($t1 $(, $ts)*);
				fn next(&mut self) -> Option<Self::Item> { self.iter.next() }
				fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
			}

			impl<'a, $t1: 'a $(, $ts: 'a)*> DoubleEndedIterator for Drain<'a, $t1 $(, $ts)*> {
				fn next_back(&mut self) -> Option<Self::Item> { self.iter.next_back() }
			}

			impl<'a, $t1: 'a $(, $ts: 'a)*> ExactSizeIterator for Drain<'a, $t1 $(, $ts)*> {
				fn len(&self) -> usize { self.iter.len() }
			}

			impl<'a, $t1: 'a $(, $ts: 'a)*> Drop for Drain<'a, $t1 $(, $ts)*> {
				fn drop(&mut self) {
					self.for_each(drop);

					if self.tail_len > 0 {
						unsafe {
							let source_soa = self.soa.as_mut();
							let start = source_soa.len();
							let tail = self.tail_start;
							if tail != start {
								{
									let src = source_soa.$t1.as_ptr().add(tail);
									let dst = source_soa.$t1.as_ptr().add(start);
									copy(src, dst, self.tail_len);
								}

								$({
									let src = source_soa.$ts.as_ptr().add(tail);
									let dst = source_soa.$ts.as_ptr().add(start);
									copy(src, dst, self.tail_len);
								})*
							}
							source_soa.len = start + self.tail_len;
						}
					}
				}
			}

			impl<$t1: Sized $(, $ts: Sized)*> Drop for $name<$t1 $(, $ts)*> {
				fn drop(&mut self) {
					self.clear(); // Drop owned items
					self.dealloc()
				}
			}


			impl<$t1: Clone + Sized $(, $ts: Clone + Sized)*> Clone for $name<$t1 $(, $ts)*> {
				fn clone(&self) -> Self {
					let mut result = Self::with_capacity(self.len);

					unsafe {
						for i in 0..self.len {
							// We do all the cloning first, then the ptr writing and length update
							// to ensure drop on panic in case any clone panics. If we write to early,
							// then the soa will not drop the most recently written item.
							// TODO: (Performance) - It may be better to do each slice individually,
							// but we'll need some kind of intermediate struct to handle drop before
							// everything is put into the Soa.
							let $t1 = (&*(self.$t1.as_ptr().add(i))).clone();
							$(let $ts = (&*(self.$ts.as_ptr().add(i))).clone();)*
							write(result.$t1.as_ptr().add(i), $t1);
							$(write(result.$ts.as_ptr().add(i), $ts);)*;

							result.len = i + 1;
						}
					}
					result
				}
			}

			impl<$t1: Sized $(, $ts: Sized)*> Default for $name<$t1 $(, $ts)*> {
				fn default() -> Self { Self::new() }
			}
		}
	};
}

soa!(Soa2, T1, T2);
soa!(Soa3, T1, T2, T3);
soa!(Soa4, T1, T2, T3, T4);
soa!(Soa5, T1, T2, T3, T4, T5);
soa!(Soa6, T1, T2, T3, T4, T5, T6);
soa!(Soa7, T1, T2, T3, T4, T5, T6, T7);
soa!(Soa8, T1, T2, T3, T4, T5, T6, T7, T8);

#[cfg(test)]
mod tests {
    use super::{Soa2::Soa2, Soa3::Soa3};
    use testdrop::TestDrop;

	fn assert_all_dropped(td: &TestDrop) {
		assert_eq!(td.num_dropped_items(), td.num_tracked_items());
	}

    #[test]
    fn layouts_do_not_overlap() {
        // Trying with both (small, large) and (large, small) to ensure nothing bleeds into anything else.
        // This verifies we correctly chunk the slices from the larger allocations.
        let mut soa_ab = Soa2::new();
        let mut soa_ba = Soa2::new();

        fn ab(v: usize) -> (u8, f64) {
            (v as u8, 200.0 + ((v as f64) / 200.0))
        }

        fn ba(v: usize) -> (f64, u8) {
            (15.0 + ((v as f64) / 16.0), (200 - v) as u8)
        }

        // Combined with the tests inside, also verifies that we are copying the data on grow correctly.
        for i in 0..100 {
            soa_ab.push(ab(i));
            let (a, b) = soa_ab.slices();
            assert_eq!(i + 1, a.len());
            assert_eq!(i + 1, b.len());
            assert_eq!(ab(0).0, a[0]);
            assert_eq!(ab(0).1, b[0]);
            assert_eq!(ab(i).0, a[i]);
            assert_eq!(ab(i).1, b[i]);

            soa_ba.push(ba(i));
            let (b, a) = soa_ba.slices();
            assert_eq!(i + 1, a.len());
            assert_eq!(i + 1, b.len());
            assert_eq!(ba(0).0, b[0]);
            assert_eq!(ba(0).1, a[0]);
            assert_eq!(ba(i).0, b[i]);
            assert_eq!(ba(i).1, a[i]);
        }
    }

    #[test]
    fn sort() {
        let mut soa = Soa3::new();

        soa.push((3, 'a', 4.0));
        soa.push((1, 'b', 5.0));
        soa.push((2, 'c', 6.0));

        soa.sort_unstable_by(|(a1, _, _), (a2, _, _)| a1.cmp(a2));

        assert_eq!(soa.index(0), (&1, &('b'), &5.0));
        assert_eq!(soa.index(1), (&2, &('c'), &6.0));
        assert_eq!(soa.index(2), (&3, &('a'), &4.0));
    }

    #[test]
    fn drops() {
        let td = TestDrop::new();
        let (id, item) = td.new_item();
        {
            let mut soa = Soa2::new();
            soa.push((1.0, item));

            // Did not drop when moved into the soa
            td.assert_no_drop(id);

            // Did not drop through resizing the soa.
            for _ in 0..50 {
                soa.push((2.0, td.new_item().1));
            }
            td.assert_no_drop(id);
        }
        // Dropped with the soa
        td.assert_drop(id);

		assert_all_dropped(&td);
    }


    #[test]
    fn clones() {
        let mut src = Soa2::new();
        src.push((1.0, 2.0));
        src.push((3.0, 4.0));

        let dst = src.clone();
        assert_eq!(dst.len(), 2);
        assert_eq!(dst.index(0), (&1.0, &2.0));
        assert_eq!(dst.index(1), (&3.0, &4.0));
    }

	#[test]
	fn insert() {
		let mut src = Soa2::new();
		src.insert(0, (1, 2));
		src.insert(0, (3, 4));
		src.insert(1, (4, 5));
		assert_eq!(src.index(0), (&3, &4));
		assert_eq!(src.index(1), (&4, &5));
		assert_eq!(src.index(2), (&1, &2));
	}

	#[test]
	fn remove() {
		let mut src = Soa2::new();
		src.push((1, 2));
		src.push((3, 4));
		assert_eq!(src.remove(0), (1, 2));
		assert_eq!(src.remove(0), (3, 4));
		assert_eq!(src.len(), 0);
	}

	#[test]
	fn drain() {

		let td = TestDrop::new();
        let (id, item) = td.new_item();
        let mut soa = Soa2::new();
        soa.push((1.0, item));
		let (_id2, item2) = td.new_item();
		soa.push((1.0, item2));

		let drain = soa.drain(..1);
		// Not dropped when moved into drain
		td.assert_no_drop(id);

		// drain yields only the items in the range
		let mut count = 0;
		for drained in drain {
			count += 1;
			assert_eq!(drained.0, 1.0);
		}
		assert_eq!(1, count);


		// Item was dropped
        td.assert_drop(id);

		// Unaltered item left
		assert_eq!(, );

		// Not doubly dropped
		drop(soa);

		assert_all_dropped(&td);
	}
}
