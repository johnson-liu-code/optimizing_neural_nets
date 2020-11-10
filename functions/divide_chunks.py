

### For individuals with multiple layers, this function chops up
###     the chromosome vector into chunks for each layer.
### Don't know how this function works. Got this from stackoverflow.

def divide_chunks(my_list, n):
    chunks = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n )]
    return chunks

