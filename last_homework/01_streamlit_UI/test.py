# テストケースの作成
def test_student_function(app_function):
    test_cases = [
        (1, 2),  # 入力 1 → 期待値 2
        (2, 4),  # 入力 2 → 期待値 4
        (3, 6),  # 入力 3 → 期待値 6
        (4, 8),  # 入力 4 → 期待値 8
        (5, 10), # 入力 5 → 期待値 10
    ]
    count = 0
    result = []
    for inp, expected in test_cases:
        if app_function(inp) == expected:
            count +=1
        else:
            result.append((inp,expected,app_function(inp))) 
    return count,result
