def num2_str(number: float, money: bool = False, zero: bool = True) -> str:
    dot = "."
    separator = " "
    minus = "مىنۇس"
    dot_str = "پۈتۈن"
    yuan = "يۈەن"
    jiao = "مو"
    fen = "پۇڭ"
    one = ["نۆل", "بىر", "ئىككى", "ئۈچ", "تۆت", "بەش", "ئالتە", "يەتتە", "سەككىز", "توققۇز"]
    two = ["ئون", "يىگىرمە", "ئوتتۇز", "قىرىق", "ئەللىك", "ئاتمىش", "يەتمىش", "سەكسەن", "توقسان"]
    more = ["يۈز", "مىڭ", "مىليون", "مىليارد", "تىرىللىيون"]
    decimal_more = ["ئوندا", "يۈزدە", "مىڭدە", "ئون مىڭدە", "يۈز مىڭدە", "مىليوندا", "ئون مىليوندا", "يۈز مىليوندا", "مىلياردتا", "ئون مىلياردتا", "يۈز مىلياردتا", "تىرىللىيوندا", "ئون تىرىللىيوندا", "يۈز تىرىللىيوندا"]

    source = str(number)
    split = source.split(dot)

    integer = split[0]
    decimal = split[1] if len(split) == 2 else ""

    result = ""
    if int(integer) < 0:
        result += minus + separator
        integer = integer[1:].strip()

    if len(integer) == 1:
        result += one[int(integer)] if integer != "0" or zero or len(decimal) > 0 else ""
    elif len(integer) == 2:
        first = int(integer[0])
        second = int(integer[1])
        result += two[first - 1] + separator + num2_str(second, False)
    else:
        index = 1 if len(integer) == 3 else len(integer) % 3
        index = 3 if index == 0 else index

        more_index = (len(integer) - 1) // 3
        if more_index > len(more) - 1:
            raise ValueError("Number is too large")

        first = int(integer[:index])
        result += num2_str(first) + separator + more[more_index] + separator + num2_str(int(integer[index:]), False)

    if len(decimal) > len(decimal_more):
        raise ValueError("Number is too large")
    elif len(decimal) > 0 and not money:
        result += separator + dot_str + separator + decimal_more[len(decimal) - 1] + separator + num2_str(int(decimal), False)
    elif money:
        result += separator + yuan
        num1 = int(decimal[0]) if len(decimal) > 0 else 0
        num2 = int(decimal[1]) if len(decimal) > 1 else 0
        if num1 > 0:
            result += separator + num2_str(num1) + separator + jiao
        if num2 > 0:
            result += separator + num2_str(num2) + separator + fen

    return result.strip()


if __name__ =="__main__":
    print(num2_str(123456789))
    print(num2_str(123.456789))
    print(num2_str(1.10, True))
    print(num2_str(1.01, True))
    print(num2_str(1.12, True))
