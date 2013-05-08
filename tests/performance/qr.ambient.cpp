#include "params.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE( test, T, test_types){
    typedef ambient::dim2 dim;
    typedef alps::numeric::matrix<typename T::value_type> sMatrix;
    typedef ambient::numeric::tiles<ambient::numeric::matrix<typename T::value_type> > pMatrix;

    size_t x = get_input_x<T>();
    size_t y = get_input_y<T>();
    size_t nthreads = get_input_threads<T>();

    pMatrix pA(x, y);
    pMatrix pQ(x, y);
    pMatrix pR(x, y);

    sMatrix sA(x, y);
    sMatrix sQ(x, y);
    sMatrix sR(x, y);

    generate(pA);
    sA = cast<sMatrix>(pA);

    ambient::sync();
    qr(sA, sQ, sR);
    qr(pA, pQ, pR); 

    ambient::timer time("ambient");
    time.begin();
    ambient::sync();
    time.end();

    report(time, GFlopsGemm, x, y, nthreads);
    ambient::sync();

    BOOST_CHECK(sQ == pQ);
    BOOST_CHECK(sR == pR);
}
