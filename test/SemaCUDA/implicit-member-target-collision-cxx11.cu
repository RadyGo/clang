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

void hostfoo4() {
  C3_with_collision c; // expected-error {{implicitly-deleted default constructor}}
}
