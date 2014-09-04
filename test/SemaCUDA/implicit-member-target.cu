// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

//------------------------------------------------------------------------------
// Test 1: infer default ctor to be host.

struct A1_with_host_ctor {
  A1_with_host_ctor() {}
};

// The implicit default constructor is inferred to be host because it only needs
// to invoke a single host constructor (A1_with_host_ctor's). So we'll encounter
// an error when calling it from a __device__ function, but not from a __host__
// function.
struct B1_with_implicit_default_ctor : A1_with_host_ctor {
};

// expected-note@-3 {{call to __host__ function from __device__}}
// expected-note@-4 {{requires 1 argument}}

void hostfoo() {
  B1_with_implicit_default_ctor b;
}

__device__ void devicefoo() {
  B1_with_implicit_default_ctor b; // expected-error {{no matching constructor}} 
}

//------------------------------------------------------------------------------
// Test 2: infer default ctor to be device.

struct A2_with_device_ctor {
  __device__ A2_with_device_ctor() {}
};

struct B2_with_implicit_default_ctor : A2_with_device_ctor {
};

// expected-note@-3 {{call to __device__ function from __host__}}
// expected-note@-4 {{requires 1 argument}}

void hostfoo2() {
  B2_with_implicit_default_ctor b;  // expected-error {{no matching constructor}} 
}

__device__ void devicefoo2() {
  B2_with_implicit_default_ctor b;
}

//------------------------------------------------------------------------------
// Test 3: infer copy ctor

struct A3_with_device_ctors {
  __host__ A3_with_device_ctors() {}
  __device__ A3_with_device_ctors(const A3_with_device_ctors&) {}
};

struct B3_with_implicit_ctors : A3_with_device_ctors {
};

// expected-note@-3 {{(the implicit copy constructor) not viable: call to __device__ function from __host__}}
// expected-note@-4 {{requires 0 arguments}}

void hostfoo3() {
  B3_with_implicit_ctors b;  // this is OK because the inferred default ctor
                             // here is __host__
  B3_with_implicit_ctors b2 = b; // expected-error {{no matching constructor}} 

}

//------------------------------------------------------------------------------
// Test 4: infer default ctor from a field, not a base

struct A4_with_host_ctor {
  A4_with_host_ctor() {}
};

struct B4_with_implicit_default_ctor {
  A4_with_host_ctor field;
};

// expected-note@-4 {{call to __host__ function from __device__}}
// expected-note@-5 {{requires 1 argument}}

void hostfoo4() {
  B4_with_implicit_default_ctor b;
}

__device__ void devicefoo4() {
  B4_with_implicit_default_ctor b; // expected-error {{no matching constructor}} 
}
