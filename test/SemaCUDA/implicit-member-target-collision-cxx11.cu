// RUN: %clang_cc1 -std=gnu++11 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

//------------------------------------------------------------------------------
// Test 1: collision between two bases

struct A1_with_host_ctor {
  A1_with_host_ctor() {}
};

struct B1_with_device_ctor {
  __device__ B1_with_device_ctor() {}
};

struct C1_with_collision : A1_with_host_ctor, B1_with_device_ctor {
};

// expected-error@-3 {{implicit default constructor inferred target collision: call to both __host__ and __device__ members}}

void hostfoo1() {
  C1_with_collision c; // expected-error {{implicitly-deleted default constructor}}
}

//------------------------------------------------------------------------------
// Test 2: collision between two fields

struct C2_with_collision {
  A1_with_host_ctor aa;
  B1_with_device_ctor bb;
};

// expected-error@-5 {{implicit default constructor inferred target collision: call to both __host__ and __device__ members}}

void hostfoo2() {
  C2_with_collision c; // expected-error {{implicitly-deleted default constructor}}
}

//------------------------------------------------------------------------------
// Test 3: collision between a field and a base

struct C3_with_collision : A1_with_host_ctor {
  B1_with_device_ctor bb;
};

// expected-error@-4 {{implicit default constructor inferred target collision: call to both __host__ and __device__ members}}

void hostfoo3() {
  C3_with_collision c; // expected-error {{implicitly-deleted default constructor}}
}

//------------------------------------------------------------------------------
// Test 4: collision on resolving a copy ctor

struct A4_with_host_copy_ctor {
  A4_with_host_copy_ctor() {}
  A4_with_host_copy_ctor(const A4_with_host_copy_ctor&) {}
};

struct B4_with_device_copy_ctor {
  B4_with_device_copy_ctor() {}
  __device__ B4_with_device_copy_ctor(const B4_with_device_copy_ctor&) {}
};

struct C4_with_collision : A4_with_host_copy_ctor, B4_with_device_copy_ctor {
};

// expected-error@-3 {{implicit copy constructor inferred target collision}}

void hostfoo4() {
  C4_with_collision c;
  C4_with_collision c2 = c; // expected-error {{implicitly-deleted copy constructor}}
}

//------------------------------------------------------------------------------
// Test 5: collision on resolving a move ctor

struct A5_with_host_move_ctor {
  A5_with_host_move_ctor() {}
  A5_with_host_move_ctor(A5_with_host_move_ctor&&) {}
// expected-note@-1 {{copy constructor is implicitly deleted because 'A5_with_host_move_ctor' has a user-declared move constructor}}
};

struct B5_with_device_move_ctor {
  B5_with_device_move_ctor() {}
  __device__ B5_with_device_move_ctor(B5_with_device_move_ctor&&) {}
};

struct C5_with_collision : A5_with_host_move_ctor, B5_with_device_move_ctor {
};
// expected-note@-2 {{deleted}}

void hostfoo5() {
  C5_with_collision c;
  // What happens here:
  // This tries to find the move ctor. Since the move ctor is deleted due to
  // collision, it then looks for a copy ctor. But copy ctors are implicitly
  // deleted when move ctors are declared explicitly.
  C5_with_collision c2(static_cast<C5_with_collision&&>(c)); // expected-error {{call to implicitly-deleted}}
}
