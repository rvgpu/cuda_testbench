function(add_cudatb_basic fname)
    add_cudatb(cudatb_basic_${fname} ${fname}.cu)
endfunction()

add_cudatb_basic(array)
add_cudatb_basic(thread_num)
