import re


class ExpressionParser:
    def __init__(self, expression: str):
        self.expression = expression
        self.tokens = []
        self.current_token_index = 0

    def tokenize(self):
        token_pattern = re.compile(
            r"\d+|d|\+|\-|\*|//\^|//|\(|\)|A|D|{|}|:|,|\[.+?\]_\d+"
        )
        self.tokens = token_pattern.findall(self.expression)


# Complex expression covering various cases
complex_expression = (
    "A(2d6, d{1:2, 2, 4:2, 5:2}) * D(1d6, 4)d(1d2) - "
    "40//8d[1,1]_7 + 3 * (d{2, 3:3, 6} + d6) - "
    "5d{1:3, 3, 7:4} + 10 //^ 2d4 - "
    "A(d6, d{1, 5:2, 6}) + "
    "D(2 * (1 + 3d4), 1d20) //^ 5 - "
    "3d{2d6 * 1d8, 5:2, 2d6 + 3d{1,2,3} : 4}"
)

# Expected tokens
expected_tokens = [
    "A",
    "(",
    "2",
    "d",
    "6",
    ",",
    "d",
    "{",
    "1",
    ":",
    "2",
    ",",
    "2",
    ",",
    "4",
    ":",
    "2",
    ",",
    "5",
    ":",
    "2",
    "}",
    ")",
    "*",
    "D",
    "(",
    "1",
    "d",
    "6",
    ",",
    "4",
    ")",
    "d",
    "(",
    "1",
    "d",
    "2",
    ")",
    "-",
    "40",
    "//",
    "8",
    "d",
    "[1,1]_7",
    "+",
    "3",
    "*",
    "(",
    "d",
    "{",
    "2",
    ",",
    "3",
    ":",
    "3",
    ",",
    "6",
    "}",
    "+",
    "d",
    "6",
    ")",
    "-",
    "5",
    "d",
    "{",
    "1",
    ":",
    "3",
    ",",
    "3",
    ",",
    "7",
    ":",
    "4",
    "}",
    "+",
    "10",
    "//^",
    "2",
    "d",
    "4",
    "-",
    "A",
    "(",
    "d",
    "6",
    ",",
    "d",
    "{",
    "1",
    ",",
    "5",
    ":",
    "2",
    ",",
    "6",
    "}",
    ")",
    "+",
    "D",
    "(",
    "2",
    "*",
    "(",
    "1",
    "+",
    "3",
    "d",
    "4",
    ")",
    ",",
    "1",
    "d",
    "20",
    ")",
    "//^",
    "5",
    "-",
    "3",
    "d",
    "{",
    "2",
    "d",
    "6",
    "*",
    "1",
    "d",
    "8",
    ",",
    "5",
    ":",
    "2",
    ",",
    "2",
    "d",
    "6",
    "+",
    "3",
    "d",
    "{",
    "1",
    ",",
    "2",
    ",",
    "3",
    "}",
    ":",
    "4",
    "}",
]

# Tokenize
parser = ExpressionParser(complex_expression)
parser.tokenize()

# Print tokens
print(parser.tokens)

# Assert test
try:
    assert (
        parser.tokens == expected_tokens
    ), f"Failed: expected {expected_tokens}, got {parser.tokens}"
except AssertionError as e:
    print(e)
else:
    print("Passed")
