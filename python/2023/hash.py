class HashSloution:
    # 1015 https://leetcode.cn/problems/smallest-integer-divisible-by-k/
    # (a+b)%c = (a%c+b%c)%c
    # (a*b)%c = ((a%c)*(b%c))%c
    def smallestRepunitDivByK(self, k: int) -> int:
        # 2 5
        if k % 2 == 0 or k % 5 == 0:
            return -1
        keySet = set()

        # modNew = (i*10+1)%k = (i%k * 10%k + 1%k)%k = (modOld*10%k+1%k)%k
        mod = 1 % k
        if mod == 0:
            return 1
        keySet.add(mod)
        while True:
            mod = (mod * 10 % k + 1 % k) % k
            if mod == 0:
                return len(keySet) + 1
            if mod in keySet:
                return -1
            keySet.add(mod)


if __name__ == '__main__':
    S = HashSloution()
    print(S.smallestRepunitDivByK(1))
