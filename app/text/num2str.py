def num2str(number: float, money: bool = False, zero: bool = True) -> str:
    """
    将数字转换为维吾尔语的字符串表示形式。

    :param number: 要转换的数字（浮点数）
    :param money: 如果为 True，则以货币格式显示（例如：一元一角）
    :param zero: 是否显示零（默认显示）
    :return: 维吾尔语的数字字符串
    """
    # 定义中文数字词汇和单位
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
    decimal_more = ["ئوندا", "يۈزدە", "مىڭدە", "ئون مىڭدە", "يۈز مىڭدە", "مىليوندا", "ئون مىليوندا", "يۈز مىليوندا",
                    "مىلياردتا", "ئون مىلياردتا", "يۈز مىلياردتا", "تىرىللىيوندا", "ئون تىرىللىيوندا",
                    "يۈز تىرىللىيوندا"]

    # 将数字转换为字符串并分割整数部分和小数部分
    source = str(number)
    split = source.split(dot)

    integer = split[0]  # 整数部分
    decimal = split[1] if len(split) == 2 else ""  # 小数部分

    result = ""
    # 处理负数
    if int(integer) < 0:
        result += minus + separator
        integer = integer[1:].strip()

    # 处理整数部分
    if len(integer) == 1:
        result += one[int(integer)] if integer != "0" or zero or len(decimal) > 0 else ""
    elif len(integer) == 2:
        first = int(integer[0])
        second = int(integer[1])
        result += two[first - 1] + separator + num2str(second, False)
    else:
        index = 1 if len(integer) == 3 else len(integer) % 3
        index = 3 if index == 0 else index

        more_index = (len(integer) - 1) // 3
        if more_index > len(more) - 1:
            raise ValueError("Number is too large")

        first = int(integer[:index])
        result += num2str(first) + separator + more[more_index] + separator + num2str(int(integer[index:]), False)

    # 处理小数部分
    if len(decimal) > len(decimal_more):
        raise ValueError("Number is too large")
    elif len(decimal) > 0 and not money:
        result += separator + dot_str + separator + decimal_more[len(decimal) - 1] + separator + num2str(int(decimal),
                                                                                                         False)
    elif money:
        result += separator + yuan
        num1 = int(decimal[0]) if len(decimal) > 0 else 0
        num2 = int(decimal[1]) if len(decimal) > 1 else 0
        if num1 > 0:
            result += separator + num2str(num1) + separator + jiao
        if num2 > 0:
            result += separator + num2str(num2) + separator + fen

    return result.strip()


if __name__ == "__main__":
    print(num2str(123456789))  # 打印: 维吾尔语表示形式的 123456789
    print(num2str(123.456789))  # 打印: 维吾尔语表示形式的 123.456789
    print(num2str(1.10, True))  # 打印: 维吾尔语表示形式的 1.10（货币格式）
    print(num2str(1.01, True))  # 打印: 维吾尔语表示形式的 1.01（货币格式）
    print(num2str(1.12, True))  # 打印: 维吾尔语表示形式的 1.12（货币格式）
