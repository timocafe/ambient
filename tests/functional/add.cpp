#include "utils/testing.hpp"

TEST_CASE("Matrix sum is computed", "[add]")
{
    matrix<double> A(TEST_M, TEST_N);
    matrix<double> B(TEST_M, TEST_N);
    matrix<double> C(TEST_M, TEST_N);

    generate(A);
    generate(B);

    C = A + B;
    C -= B;

    REQUIRE((C == A));
}

TEST_CASE("Matrix scale is computed", "[scale]")
{
    matrix<double> A(64, 64, 1.);
    matrix<double> B(64, 64, 2.);
    matrix<double> C(64, 64, 2.);


    C = A * B;

}

TEST_CASE("Matrix gemms is computed", "[scale]")
{
    matrix<double> A(TEST_M, TEST_N, 1.);
    matrix<double> B(TEST_M, TEST_N, 2.);

    A *= 2;

    REQUIRE((B == A));

}