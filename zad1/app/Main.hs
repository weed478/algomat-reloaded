-- UTILS

type Mat a = [[a]]

mopm :: Num a => (a -> a -> a) -> Mat a -> Mat a -> Mat a
mopm = zipWith . zipWith

maddm :: Num a => Mat a -> Mat a -> Mat a
maddm = mopm (+)

msubm :: Num a => Mat a -> Mat a -> Mat a
msubm = mopm (-)

mslice :: Int -> Int -> Int -> Int -> Mat a -> Mat a
mslice i0 i1 j0 j1 = map (take (j1 - j0) . drop j0) . take (i1 - i0) . drop i0

qsplit :: Mat a -> (Mat a, Mat a, Mat a, Mat a)
qsplit a = (a11, a12, a21, a22)
    where
        n = length a
        m = n `div` 2
        a11 = mslice 0 m 0 m a
        a12 = mslice 0 m m n a
        a21 = mslice m n 0 m a
        a22 = mslice m n m n a

qmerge :: (Mat a, Mat a, Mat a, Mat a) -> Mat a
qmerge (a11, a12, a21, a22) = zipWith (++) a11 a12 ++ zipWith (++) a21 a22

-- !UTILS

-- BINET

binet :: Num a => Mat a -> Mat a -> Mat a
binet [[a]] [[b]] = [[a * b]]
binet a b = c
    where
        (a11, a12, a21, a22) = qsplit a
        (b11, b12, b21, b22) = qsplit b

        c11 = binet a11 b11 `maddm` binet a12 b21
        c12 = binet a11 b12 `maddm` binet a12 b22
        c21 = binet a21 b11 `maddm` binet a22 b21
        c22 = binet a21 b12 `maddm` binet a22 b22

        c = qmerge (c11, c12, c21, c22)

-- !BINET

-- STRASSEN

strassen :: Num a => Mat a -> Mat a -> Mat a
strassen [[a]] [[b]] = [[a * b]]
strassen a b = if n <= 16 then binet a b else c
    where
        n = length a

        (a11, a12, a21, a22) = qsplit a
        (b11, b12, b21, b22) = qsplit b

        p1 = strassen (a11 `maddm` a22) (b11 `maddm` b22)
        p2 = strassen (a21 `maddm` a22) b11
        p3 = strassen a11 (b12 `msubm` b22)
        p4 = strassen a22 (b21 `msubm` b11)
        p5 = strassen (a11 `maddm` a12) b22
        p6 = strassen (a21 `msubm` a11) (b11 `maddm` b12)
        p7 = strassen (a12 `msubm` a22) (b21 `maddm` b22)

        c11 = p1 `maddm` p4 `msubm` p5 `maddm` p7
        c12 = p3 `maddm` p5
        c21 = p2 `maddm` p4
        c22 = p1 `msubm` p2 `maddm` p3 `maddm` p6

        c = qmerge (c11, c12, c21, c22)

-- !STRASSEN

main :: IO ()
main = do
    -- Program reads stdin "n a00 a01 a01 ... b00 b01 b02 ...", where n is the size of the matrix
    args <- words <$> getContents
    let n = read $ head args :: Int
    let avec = map read $ take (n * n) $ tail args :: [Double]
    let bvec = map read $ drop (n * n) $ tail args :: [Double]
    let a = map (take n) $ take n $ iterate (drop n) avec
    let b = map (take n) $ take n $ iterate (drop n) bvec
    let c = strassen a b
    -- Output is the flattened matrix with elements separated by spaces
    putStrLn $ unwords $ map (unwords . map show) c
