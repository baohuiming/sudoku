# 作者：bjtu-bhm
# 12/31/2021 OpenCV识别手写数独核心代码
# 需要有利用sklearn自行训练好的svm,pca模型
# load_models()作用是全局载入模型
# svm_printed是识别印刷体数字的模型
# MODELS = [SVM, PCA, SVM_PRINTED] # 分别是sklearn训练好的模型对象
def image_to_grid(filename) -> 'json':
    try:
        def deskew(img):
            SZ = 28
            affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR
            m = cv.moments(img)
            if abs(m['mu02']) < 1e-2:
                return img.copy()
            skew = m['mu11'] / m['mu02']
            M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
            img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
            return img

        if str(MODELS) == '[]':
            load_models()
        svm, pca, svm_printed = MODELS
        img_origin = cv.imread(filename)
        # 获取文件大小
        sz = os.path.getsize(filename)

        # 缩小图片
        width, height = img_origin.shape[1], img_origin.shape[0]
        img_origin = cv.resize(img_origin, (500, int(500 / width * height)))
        img = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY)

        # 二值化
        blur = cv.GaussianBlur(img, (3, 3), 0)
        ret, threshold = cv.threshold(blur, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        # cv.imshow('threshold', threshold)
        # 找到轮廓
        contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # 计算最大面积的轮廓
        area = [cv.contourArea(cnt) for cnt in contours]
        idx = area.index(max(area))
        # 获取四个角点
        points = cv.approxPolyDP(contours[idx], 5, True)  # 顺序不一定
        # 把角点按照顺序排列
        points = sorted(points, key=lambda x: math.pow(x[0][0], 2) + math.pow(x[0][1], 2))
        if points[1][0][0] < points[2][0][0]:  # 右上角和左下角互换
            points[1], points[2] = points[2], points[1]
        # 透视变换
        pts_origin = np.float32(points)
        pts_result = np.float32([[0, 0], [252, 0], [0, 252], [252, 252]])
        M = cv.getPerspectiveTransform(pts_origin, pts_result)
        dst = cv.warpPerspective(threshold, M, (252, 252))
        ori = cv.warpPerspective(img_origin, M, (252, 252))

        # 找出每个小方格
        contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(ori,contours,-1,(255,0,0),3)
        # 创建数独棋盘
        grid = np.zeros((9, 9, 1), dtype=int)
        # 找出每个数字

        for idx, cnt in enumerate(contours):
            x, y, w, h = cv.boundingRect(cnt)
            # cv.rectangle(ori, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 12 < h < 29 and 1 < w < 29 and h * w < 26 * 26:
                cv.rectangle(ori, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cell = dst[y:y + h, x:x + w]
                cell = cv.resize(cell, (round(20 / h * w), 20), cv.INTER_CUBIC)
                w, h = cell.shape[::-1]
                w1 = 28 - w
                h1 = 28 - h
                left, top = round(w1 / 2), round(h1 / 2)
                right, bottom = w1 - left, h1 - top

                cell = cv.copyMakeBorder(cell, top=top, bottom=bottom, left=left, right=right,
                                         borderType=cv.BORDER_CONSTANT,
                                         value=(0, 0, 0))
                if sz > 0.9 * 1024 * 1024:  # 文件大于0.9MB，调整文字方向
                    cell = deskew(cell)
                # cv.imshow(str(idx), cell)
                cell = np.reshape(cell, 784)  # 变成一行
                if cell.sum() < 255 * 9:  # 小于9个像素直接为0
                    grid[int(x / 28)][int(y / 28)] = 0
                else:
                    arr = cell.tolist()
                    if sz > 0.9 * 1024 * 1024:  # 文件大于0.9MB，用手写识别模型
                        arr = pca.transform([arr])
                        num = svm.predict(arr)
                    else:
                        num = svm_printed.predict([arr])
                    grid[int(x / 28)][int(y / 28)] = num[0]

        arr = grid.T[0].reshape(81).tolist()
        res = {'MessageType': 'success', 'Title': '提示', 'Content': arr}
        return jsonify(res)

    except Exception as err:
        if functions.is_linux():
            with open('sudoku_error.log', mode='a') as fp:
                fp.write(str(err))
        else:
            print(str(err))
        return jsonify({'MessageType': 'error', 'Title': '错误', 'Content': '图像识别失败'})
