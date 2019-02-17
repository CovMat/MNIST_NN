from struct import unpack

def read_file( fname ):

    with open( fname, "rb" ) as fin:

        # read the "magic number (MSB first)"
        f4bytes = fin.read(4)
        ( magic_num, ) = unpack( ">i", f4bytes )

        # read the number of items or images
        f4bytes = fin.read(4)
        ( total_num, ) = unpack( ">i", f4bytes )

        # read number of rows and columns for image file
        row_num = 1
        col_num = 1
        if (magic_num == 2051) :
            f8bytes = fin.read(8)
            ( row_num, col_num ) = unpack( ">2i", f8bytes )

        # read content of file
        fbytes = fin.read( total_num*row_num*col_num )
        fcontent = unpack( "B" * (total_num*row_num*col_num), fbytes )

    fin.close()
    return total_num, row_num, col_num, fcontent
