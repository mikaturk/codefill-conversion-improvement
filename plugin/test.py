def count_words(input_str):
    words = input_str.split()
    return len(words)

def count_chars_in_words(char, words):
    count = 0
    for c in words:
        if c == char:
            count += 1
    return

if __name__ == '__main__':
    string = "hello how are you"
    print("words in string: ", count_words(string))
    print("occurences of 'o' in string: ", count_chars_in_words('o', string))