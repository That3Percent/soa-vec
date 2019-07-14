
 Soa2, Soa3, ..SoaN are generic collections with an API similar to that of a Vec of tuples but which store
 the data laid out as a separate slice per field. The advantage of this layout is that when
 iterating over the data only a subset need be loaded from RAM.

 This approach is common to game engines, and entity component systems in particular but is
 applicable anywhere that cache coherency and memory bandwidth are important for performance.


 # Example
 ```rust
 # use soa_vec::Soa3;
 /// Some 'entity' data.
 # #[derive(Copy, Clone)]
 struct Position { x: f64, y: f64 }
 # #[derive(Copy, Clone)]
 struct Velocity { dx: f64, dy: f64 }
 struct ColdData { /* Potentially many fields omitted here */ }

 # use std::ops::Add;
 # impl Add<Velocity> for Position { type Output=Self; fn add(self, other: Velocity) -> Self { Self { x: self.x + other.dx, y: self.y + other.dy } } }
 // Create a vec of entities
 let mut entities = Soa3::new();
 entities.push((Position {x: 1.0, y: 2.0}, Velocity { dx: 0.0, dy: 0.5 }, ColdData {}));
 entities.push((Position {x: 0.0, y: 2.0}, Velocity { dx: 0.5, dy: 0.5 }, ColdData {}));

 // Update entities. This loop only loads position and velocity data, while skipping over
 // the ColdData which is not necessary for the physics simulation.
 let (positions, velocities, _cold) = entities.iters_mut();
 for (position, velocity) in positions.zip(velocities) {
 	*position = *position + *velocity;
 }

 // Remove an entity
 entities.swap_remove(0);

 // Sort entities by position on y axis
 // The fields are passed by reference, so velocity and cold data are not loaded
 // until such time as the items are being swapped which runs in O(N)
 # use std::cmp;
 entities.sort_unstable_by(
 	|(lh_pos, _, _), (rh_pos, _, _)| lh_pos.y.partial_cmp(&rh_pos.y).unwrap()
 );

 // See individual structs for more methods.

 ```


 # Nightly
 This crate has strict requirements for allocations and memory layout and therefore requires the following nightly features:
 * allocator_api
 * alloc_layout_extra

 # Links:
 * [Github source](https://github.com/That3Percent/soa-vec)
 * [Crate](https://crates.io/crates/soa-vec)

 # License
 [MIT](https://github.com/That3Percent/soa-vec/blob/master/LICENSE)


# Related
If you like this, you may like these other crates by Zac Burns (That3Percent)
* [second-stack](https://github.com/That3Percent/second-stack) A Rust memory allocator for large slices that don't escape the stack.
* [js-intern](https://github.com/That3Percent/js-intern) Stores one copy of each distinct JavaScript primitive
* [js-object](https://github.com/That3Percent/js-object) A macro for creating JavaScript objects