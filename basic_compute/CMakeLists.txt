function(add_cudatb_basic fname)
    add_cudatb(cudatb_basic_${fname} ${fname}.cu)
endfunction()

add_cudatb_basic(mul)
add_cudatb_basic(array_float)
add_cudatb_basic(array_int)
