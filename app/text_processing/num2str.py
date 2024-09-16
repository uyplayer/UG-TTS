dot = "."
dot_full = "پۈتۈن"
minus = "مىنۇس"
digits = ["نۆل", "بىر", "ئىككى", "ئۈچ", "تۆت", "بەش", "ئالتە", "يەتتە", "سەككىز", "توققۇز"]
two_units = ["ئون", "يىگىرمە", "ئوتتۇز", "قىرىق", "ئەللىك", "ئاتمىش", "يەتمىش", "سەكسەن", "توقسان"]
three_units = ["بىر يۈز", "ئىككى يۈز", "ئۈچ يۈز", "تۆت يۈز", "بەش يۈز", "ئالتە يۈز", "يەتتە يۈز", "سەككىز يۈز",
               "توققۇز يۈز"]
units = ["", "مىڭ", "مىليون", "مىليارد", "تىرىللىيون", "تىرىللىيات"]
decimal_units = ["ئوندا", "يۈزدە", "مىڭدە", "ئون مىڭدە", "يۈز مىڭدە", "مىليوندا", "ئون مىليوندا", "يۈز مىليوندا",
                 "مىلياردتا", "ئون مىلياردتا", "يۈز مىلياردتا", "تىرىللىيوندا", "ئون تىرىللىيوندا",
                 "يۈز تىرىللىيوندا"]


def num2str(number: float, integer_accepted_length=14) -> str:
    if number < 0:
        return minus + num2str(-number)
    number_str = str(number)
    if dot in number_str:
        integer_part, decimal_part = number_str.split('.')
    else:
        integer_part, decimal_part = number_str, ''
    integer = ""
    if len(integer_part) > integer_accepted_length:
        integers = split_large_number(integer_part).split(" ")
        for item in integers:
            if item.startswith('0'):
                for sub in item:
                    if sub == "0":
                        integer = integer + " " + digits[0]
                    else:
                        integer = integer + " " + digits[int(sub)]
            else:
                integer = integer + " " + integer2str(item)
    else:
        integer = integer2str(integer_part)
    if dot in number_str:
        integer = integer + " " + dot_full
    decimal = ""
    if decimal_part != "" and int(decimal_part) != 0:
        decimal = decimal2str(decimal_part)
    result = integer + " " + decimal
    return result.strip()

def split_large_number(number_str):
    parts = [number_str[max(i - 3, 0):i] for i in range(len(number_str), 0, -3)]
    return ' '.join(parts[::-1])

def convert3digit(number: str) -> str:
    if number == "0":
        return digits[0]
    result = ""
    if len(number) == 1:
        result = digits[int(number)]
    if len(number) == 2:
        if int(number[0]) !=0:
            result = two_units[int(number[0]) - 1]
        if int(number[1]) != 0:
            result = result + " " + digits[int(number[1])]
    if len(number) == 3:
        if int(number[0]) != 0:
            result = three_units[int(number[0]) - 1]
        if int(number[1]) != 0:
            result = result + " " + two_units[int(number[1]) - 1]
        if int(number[2]) != 0:
            result = result + digits[int(number[2])]
    return result.strip()


def integer2str(integer_part: str) -> str:
    group3 = [integer_part[max(i - 3, 0):i][::-1] for i in range(len(integer_part), 0, -3)]
    group3 = group3[::-1]
    group3 = [item[::-1] for item in group3]
    length = len(group3)
    result = ""
    for index, digit3 in enumerate(group3):
        con3 = convert3digit(digit3)
        more = units[length - index - 1]
        result = result + " " + con3 + " " + more
    return result.strip()


def decimal2str(decimal_part: str) -> str:
    length = len(decimal_part)
    prefix = decimal_units[length-1]
    if length > 14:
        decimal_part = decimal_part[:13]
    integer = integer2str(decimal_part)
    return prefix + " " + integer


if __name__ == '__main__':
    print(num2str(114546500008545))
    print(num2str(1145465.65646))
