AdjList_Chicago ={
1:[2, 77],
2:[1, 4, 13, 77],
3:[4, 5, 6, 77],
4:[2, 13, 14, 16, 5, 6, 3, 77],
5:[4, 14, 16, 21, 22, 7, 6, 3],
6:[3, 4, 5, 7],
7:[6, 5, 22, 24, 8],
8:[7, 24, 28, 32],
9:[10],
10:[9, 11, 12, 76],
11:[10, 12, 15],
12:[10, 11, 15, 16, 14, 13],
13:[12, 14, 4, 2],
14:[13, 12, 15, 16, 5, 4],
15:[11, 10, 17, 19, 20, 16, 14, 12],
16:[14, 12, 15, 19, 20, 21, 5, 4],
17:[76, 18, 19, 15],
18:[17, 19, 25],
19:[15, 17, 18, 25, 23, 20, 16, 15],
20:[16, 15, 19, 25, 23, 22, 21],
21:[16, 20, 22, 5],
22:[21, 20, 23, 24, 7, 5],
23:[20, 19, 25, 26, 27, 28, 24, 22],
24:[22, 23, 27, 28, 8, 7],
25:[18, 19, 20, 23, 26, 29],
26:[23, 25, 29, 27],
27:[23, 26, 29, 28, 24],
28:[24, 27, 29, 31, 33, 32, 8],
29:[26, 25, 30, 31, 28, 27],
30:[29, 56, 57, 58, 59, 31],
31:[28, 29, 30, 58, 59, 60, 34, 33],
32:[8, 28, 33],
33:[32, 28, 31, 34, 35],
34:[31, 60, 61, 37, 38, 35, 33],
35:[33, 34, 37, 38, 36],
36:[35, 38, 39],
37:[34, 60, 61, 68, 40, 38, 35],
38:[35, 34, 37, 40, 41, 39, 36],
39:[36, 38, 40, 41],
40:[38, 37, 68, 69, 42, 41, 39],
41:[39, 38, 40, 42],
42:[41, 40, 69, 43],
43:[42, 69, 45, 46],
44:[69, 71, 49, 50, 47, 45],
45:[69, 44, 47, 48, 46, 43],
46:[43, 45, 48, 51, 52],
47:[44, 49, 50, 51, 48, 45],
48:[45, 47, 50, 51, 52, 46],
49:[44, 73, 75, 53, 54, 50, 47],
50:[47, 49, 53, 54, 51, 48, 44],
51:[48, 47, 50, 54, 55, 52, 46],
52:[46, 48, 51, 55],
53:[75, 49, 50, 54],
54:[53, 50, 51, 55, 49],
55:[54, 51, 52],
56:[64, 65, 62, 57, 30],
57:[56, 62, 63, 58, 30],
58:[30, 57, 62, 63, 61, 59, 31],
59:[31, 30, 58, 61, 60],
60:[31, 59, 61, 37, 34],
61:[59, 58, 63, 67, 68, 37, 60],
62:[57, 56, 64, 65, 66, 63, 58],
63:[58, 57, 62, 65, 66, 67, 61],
64:[56, 62, 65],
65:[64, 62, 66, 70],
66:[63, 62, 65, 70, 71, 67],
67:[61, 63, 66, 70, 71, 68, 61],
68:[61, 67, 71, 69, 40, 37],
69:[40, 68, 71, 44, 45, 43, 42],
70:[65, 66, 67, 71, 72],
71:[67, 66, 70, 72, 73, 49, 44, 69, 68],
72:[70, 71, 73, 75, 74],
73:[71, 72, 75, 53, 49],
74:[72, 75],
75:[74, 72, 73, 49, 53],
76:[10, 17],
77:[1, 2, 4, 3]
}

if __name__ == "__main__":
    print(AdjList_Chicago[1])
    #check the largest degree of adjcent nodes
    #Chicago is 9
    max_len = 0
    for l in AdjList_Chicago.values():
        if len(l) > max_len:
            max_len = len(l)
    print(max_len)
