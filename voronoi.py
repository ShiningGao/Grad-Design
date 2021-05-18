# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

matplotlib.use("TkAgg")


# 初始化参数
place_x, place_y = [-1.2, 1.2], [-1.2, 1.2]
place = np.array([[place_x[0], place_y[0]],
                  [place_x[1], place_y[0]],
                  [place_x[1], place_y[1]],
                  [place_x[0], place_y[1]],
                  [place_x[0], place_y[0]]])
p_color = ['r', 'g', 'y', 'c', 'm', 'k']

rw, lw, t = 0.1, 0.178, 0.01
alpha, beta, gamma = 0.3, 5.0, 1e-3
d1, d2 = 0.1, 0.05
max_value = 999999

mode = 1
num_pursuer, p = 2, []
success_flag = False


class Line:
    def __init__(self):
        # ax + by = c
        self.a, self.b, self.c = None, None, None
        self.left, self.right = None, None

    def f(self, x):
        return (self.c - self.a * x) / self.b


class Agent:
    def __init__(self, pos, mode=0, type=None):
        self.pos = pos
        self.motion = []
        self.mode = mode

        if type == 'p':
            self.Li, self.li = -1.0, -1.0
            self.midpoint = None
            self.line = None  # line of control
            self.xi_h, self.xi_v = None, None
            self.v = None
            self.motion_bound = [1.0, 2.84]
        if type == 'e':
            if self.mode == 0:
                self.motion_bound = [1.0, np.pi]
                self.motion_reso = [0.1, 0.1]
            elif self.mode == 1:
                self.motion_bound = [1.0, 2.84]
                self.motion_reso = [0.1, 0.1]


# 位置更新
def update_pos(mode, pos, motion):
    global t

    res = np.zeros(3)
    if mode == 0:  # holonomic
        # motion = [vx, vy]
        res[0] = pos[0] + motion[0] * t
        res[1] = pos[1] + motion[1] * t
        res[2] = np.arctan2(motion[1], motion[0])
    elif mode == 1:  # non-holonomic
        # motion = [v, w]
        R = motion[0] / motion[1]
        res[0] = pos[0] - R * np.sin(pos[2]) + R * np.sin(pos[2] + motion[1] * t)
        res[1] = pos[1] + R * np.cos(pos[2]) - R * np.cos(pos[2] + motion[1] * t)
        res[2] = (pos[2] + motion[1] * t) % (2 * np.pi)

    return res


# 判断两直线相交
def intersaction(l1, l2, bound=False):
    # 默认l2存在边界约束，即线段
    A = np.array([[l1.a, l1.b],
                  [l2.a, l2.b]])
    b = np.array([[l1.c],
                  [l2.c]])

    if np.linalg.det(A) == 0:
        return False, None
    else:
        point = np.dot(np.linalg.inv(A), b).squeeze()
        if bound:
            if l2.left[0] <= point[0] <= l2.right[0]:
                if (l2.left[1] <= point[1] <= l2.right[1]) or (l2.right[1] <= point[1] <= l2.left[1]):
                    return True, point
                else:
                    return False, None
            else:
                return False, None

        return True, point


# 计算line of control的基线
def cal_baseline(p, e, bound):
    l = Line()
    xi_h = e.pos[:2] - p.pos[:2]
    midpoint = (e.pos[:2] + p.pos[:2]) / 2.0

    # 构造直线
    if p.pos[0] == e.pos[0]:
        l.a, l.b, l.c = 0, 1.0, midpoint[1]
    elif p.pos[1] == e.pos[1]:
        l.a, l.b, l.c = 1.0, 0, midpoint[0]
    else:
        k_h = xi_h[1] / xi_h[0]
        k_v = -1.0 / k_h
        l.a, l.b, l.c = k_v, -1.0, k_v * midpoint[0] - midpoint[1]

    # 设置线段端点
    for b in bound:
        flag, point = intersaction(l, b, bound=True)
        if flag:
            if l.left is None:
                l.left = point
            elif l.right is None:
                if not (l.left == point).all():
                    if l.left[0] <= point[0]:
                        l.right = point
                    else:
                        l.right = l.left
                        l.left = point

    return l


# 计算Ve的顶点集
def voronoi_set(pset, e, Ne):
    # 判断顶点是否与e都在边界的同一侧
    res, nset = [], Ne[:]
    for i, pursuer in enumerate(pset):
        flag = np.sign(e.pos[1] - pursuer.line.f(e.pos[0]))
        for n in nset:
            # 结果在0附近存在浮点数精度问题，必须讨论结果是否为0
            y = n[1] - pursuer.line.f(n[0])
            if abs(y) <= 1e-5:
                res.append(n)
            else:
                if np.sign(y) == flag:
                    res.append(n)

        # 从筛选后的点集中再进行筛选
        nset = res[:]
        if i != len(pset) - 1:
            del res[:]

    return res


# 以角度进行排序
def take_angle(element):
    return element[1]


# 逆时针排序多边形顶点集
def sort_nset(nset):
    res = [(nset[0], 0.0)]
    center = sum(nset) / len(nset)
    axis = nset[0] - center
    for i in range(1, len(nset)):
        vector = nset[i] - center
        angle = np.arccos(axis.dot(vector) / (np.linalg.norm(axis) * np.linalg.norm(vector)))
        # 角度[0, 2*pi]
        if np.cross(np.append(vector, 0), np.append(axis, 0))[2] >= 0:
            angle = 2 * np.pi - angle
        res.append((nset[i], angle))
    res.sort(key=take_angle)

    for i in range(len(res)):
        res[i] = res[i][0]

    # 计算Voronoi分区面积
    A = 0.0
    for i in range(len(res)):
        if i != len(res) - 1:
            A += res[i][0] * res[i + 1][1] - res[i + 1][0] * res[i][1]
        else:
            A += res[i][0] * res[0][1] - res[0][0] * res[i][1]
    A /= 2.0

    return A, res


# 计算line of control和追捕者速度
def cal_line_of_control(p, e, Ne):
    p.midpoint = (e.pos[:2] + p.pos[:2]) / 2.0
    p.xi_h = e.pos[:2] - p.pos[:2]
    p.xi_v = np.array([-p.xi_h[1], p.xi_h[0]])

    # 计算line of control
    flag = False
    for n in Ne:
        y = n[1] - p.line.f(n[0])
        if abs(y) <= 1e-5:
            if not flag:
                p.line.left = n
                flag = True
            else:
                p.line.right = n

    if not flag:
        p.line = None
        p.v = p.motion_bound[0] * p.xi_h / np.linalg.norm(p.xi_h)
    else:
        if p.line.left[0] >= p.line.right[0]:
            temp = p.line.left
            p.line.left = p.line.right
            p.line.right = temp

        p.Li = np.linalg.norm(p.line.left - p.line.right)
        # 交点在line of control上
        if p.line.left[0] <= p.midpoint[0] <= p.line.right[0]:
            # 判断li所在方向
            if np.dot(p.line.left - p.midpoint, p.xi_v) <= 0:
                p.li = np.linalg.norm(p.line.left - p.midpoint)
            elif np.dot(p.line.right - p.midpoint, p.xi_v) <= 0:
                p.li = np.linalg.norm(p.line.right - p.midpoint)
        # 交点不在line of control上
        else:
            d1 = np.linalg.norm(p.line.left - p.midpoint)
            d2 = np.linalg.norm(p.line.right - p.midpoint)
            if d1 >= d2:
                p.li = d1
            else:
                p.li = d2

        # 计算控制率
        Dh = -p.Li / 2.0
        Dv = (p.li ** 2 - (p.Li - p.li) ** 2) / (2.0 * np.linalg.norm(p.xi_h))
        ah = Dh / np.sqrt(Dh ** 2 + Dv ** 2)
        av = Dv / np.sqrt(Dh ** 2 + Dv ** 2)
        p.v = -(ah * p.xi_h / np.linalg.norm(p.xi_h) + av * p.xi_v / np.linalg.norm(p.xi_v)) * p.motion_bound[0]

    return p


# 计算追捕者的非完整性控制输入
def control_input(p):
    global lw, t

    # 计算期望速度方向
    v_angle = np.arctan2(p.v[1], p.v[0])
    # 转换到[0, 2pi]
    if v_angle <= 0.0:
        v_angle += 2 * np.pi

    # 计算偏差角和控制输入
    th = v_angle - p.pos[2]
    w_flag = 1
    # 控制th在[-pi, pi]以保证速度函数拥有良好性态
    # 否则v_star在2pi和-2pi附近会出现极端负值
    if th >= np.pi:
        w_flag = -1
        th = 2 * np.pi - th  # th > 0
    elif th <= -np.pi:
        th = 2 * np.pi + th  # th > 0

    # 避免th为0时分母无定义
    if abs(th) <= 1e-5:
        th += 1e-5
    v_star = np.linalg.norm(p.v) * th * np.sin(th) / (2.0 * (1.0 - np.cos(th)))
    w_star = w_flag * th / t
    if w_star > 0:
        w = min(w_star, p.motion_bound[1])
    else:
        w = max(w_star, -p.motion_bound[1])
    v = min(v_star, p.motion_bound[0] - w * lw / 2.0)

    return [v, w]


# 计算逃跑者速度
def evader_speed(p, e, ob, last_pos):
    global max_value, alpha, beta, gamma, d1, d2, t

    best_cost = max_value
    best_motion = [None, None]
    # dwa方法
    for m1 in np.arange(-e.motion_bound[0], e.motion_bound[0], e.motion_reso[0]):
        for m2 in np.arange(-e.motion_bound[1], e.motion_bound[1], e.motion_reso[1]):
            # 对采样速度进行位置更新
            if e.mode == 0:
                pos = update_pos(e.mode, e.pos, [m1 * np.cos(m2), m1 * np.sin(m2)])
            elif e.mode == 1:
                pos = update_pos(e.mode, e.pos, [m1, m2])

            # 超出地图边界
            if pos[0] >= place_x[1] or pos[0] <= place_x[0]:
                continue
            elif pos[1] >= place_y[1] or pos[1] <= place_y[0]:
                continue

            # 计算追捕者带来的cost
            cost = 0.0
            for i in range(len(p)):
                cost += alpha * np.linalg.norm(p[i].v) / np.hypot(pos[0] - p[i].pos[0], pos[1] - p[i].pos[1])
            cost /= len(p)

            min_dist = max_value
            for o in ob:
                temp_dist = np.hypot(pos[0] - o[0], pos[1] - o[1])
                if temp_dist <= min_dist:
                    min_dist = temp_dist

            # 计算墙壁带来的cost
            if min_dist <= d1:
                if min_dist <= d2:
                    cost += max_value
                else:
                    cost += beta / min_dist
            cost += gamma / (np.hypot(pos[0] - last_pos[0], pos[1] - last_pos[1]) + 1e-5)

            if cost <= best_cost:
                best_cost = cost
                best_motion = [m1, m2]

    if e.mode == 0:
        best_motion = [best_motion[0] * np.cos(best_motion[1]),
                       best_motion[0] * np.sin(best_motion[1])]

    return best_motion


# 初始化场地四个边界的直线函数
def init_place():
    lbound = Line()
    lbound.a, lbound.b, lbound.c = 1.0, 0, place_x[0]
    lbound.left, lbound.right = np.array([place_x[0], place_y[0]]), np.array([place_x[0], place_y[1]])
    rbound = Line()
    rbound.a, rbound.b, rbound.c = 1.0, 0, place_x[1]
    rbound.left, rbound.right = np.array([place_x[1], place_y[0]]), np.array([place_x[1], place_y[1]])
    bbound = Line()
    bbound.a, bbound.b, bbound.c = 0, 1.0, place_y[0]
    bbound.left, bbound.right = np.array([place_x[0], place_y[0]]), np.array([place_x[1], place_y[0]])
    ubound = Line()
    ubound.a, ubound.b, ubound.c = 0, 1.0, place_y[1]
    ubound.left, ubound.right = np.array([place_x[0], place_y[1]]), np.array([place_x[1], place_y[1]])

    obstacle, step = [], 0.1
    for i in np.arange(place_x[0], place_x[1], step):
        obstacle.append(np.array([i, place_y[0]]))
        obstacle.append(np.array([i, place_y[1]]))
    for i in np.arange(place_y[0], place_y[1], step):
        obstacle.append(np.array([place_x[0], i]))
        obstacle.append(np.array([place_x[1], i]))

    return [lbound, rbound, bbound, ubound], obstacle


bound, obstacle = init_place()
# 初始化逃跑者和追捕者
# e = Agent(np.array([(place_x[1] - place_x[0]) * np.random.rand() + place_x[0],
#                     (place_y[1] - place_y[0]) * np.random.rand() + place_y[0],
#                     2 * np.pi * np.random.rand()]),
#           mode, type='e')
# for i in range(num_pursuer):
#     while True:
#         pp = Agent(np.array([(place_x[1] - place_x[0]) * np.random.rand() + place_x[0],
#                              (place_y[1] - place_y[0]) * np.random.rand() + place_y[0],
#                              0.0]),
#                    mode, type='p')
#
#         # [0, 2*pi]
#         pp.pos[2] = np.arctan2(e.pos[1] - pp.pos[1], e.pos[0] - pp.pos[0])
#         if pp.pos[2] <= 0.0:
#             pp.pos[2] += 2 * np.pi
#
#         if np.hypot(pp.pos[0] - e.pos[0], pp.pos[1] - e.pos[1]) > rw:
#             break
#     p.append(pp)
#     del pp

e = Agent(np.array([0.0,
                    0.0,
                    -3/4 * np.pi]),
          mode, type='e')
p.append(Agent(np.array([-1.0,
                         -0.9,
                         0.0]),
               mode, type='p'))
p.append(Agent(np.array([-0.9,
                         -1.0,
                         0.0]),
               mode, type='p'))
for i in range(num_pursuer):
    p[i].pos[2] = np.arctan2(e.pos[1] - p[i].pos[1], e.pos[0] - p[i].pos[0])
    if p[i].pos[2] <= 0.0:
        p[i].pos[2] += 2 * np.pi

# 画图设置
fig = plt.figure(0)
ax = fig.add_subplot()

epos, = ax.plot(e.pos[0], e.pos[1], 'bo')
ex, ey = [], []
print("e: %.3f %.3f %.3f" % (e.pos[0], e.pos[1], e.pos[2]))

ppos, px, py = [], [], []
for i in range(num_pursuer):
    temp, = ax.plot(p[i].pos[0], p[i].pos[1], p_color[i] + 'x')
    ppos.append(temp)
    px.append([])
    py.append([])
    print("p%d: %.3f %.3f %.3f" % (i + 1, p[i].pos[0], p[i].pos[1], p[i].pos[2]))
time_text = ax.text(0.45, 1.02, '', transform=ax.transAxes)

# 画图
last_pos = np.array([e.pos[0], e.pos[1]])
def update(ii):
    global success_flag, mode, last_pos, t, num_pursuer

    # 画图
    plt.plot(place[:, 0], place[:, 1], 'k-')

    epos.set_data(e.pos[0], e.pos[1])
    ex.append(e.pos[0])
    ey.append(e.pos[1])
    ax.plot(ex, ey, 'b-', lw=0.1)

    for i in range(num_pursuer):
        ppos[i].set_data(p[i].pos[0], p[i].pos[1])
        px[i].append(p[i].pos[0])
        py[i].append(p[i].pos[1])
        ax.plot(px[i], py[i], p_color[i] + '-', lw=0.1)

    # 计算每个追捕者的基线
    for i in range(num_pursuer):
        p[i].line = cal_baseline(p[i], e, bound)

    # 计算Voronoi分区顶点和面积
    Ne = [place[0], place[1], place[2], place[3]]
    for i in range(num_pursuer):
        # 添加端点
        Ne.append(p[i].line.left)
        Ne.append(p[i].line.right)
        if i == num_pursuer - 1:
            continue
        for j in range(i + 1, num_pursuer):
            flag, point = intersaction(p[i].line, p[j].line, bound=True)
            if flag:
                Ne.append(point)  # 添加交点
    Ne = voronoi_set(p, e, Ne)
    A, Ne = sort_nset(Ne)

    dmin = max_value
    for i in range(num_pursuer):
        # 判断是否围捕成功
        d = np.linalg.norm(p[i].pos[:2] - e.pos[:2])
        if d <= dmin:
            dmin = d

        if success_flag:
            break
        if d <= rw:
            success_flag = True
            print("Success")
            break

        # 计算追捕者速度和位置更新
        p[i] = cal_line_of_control(p[i], e, Ne)
        if mode == 0:
            p[i].pos = update_pos(mode, p[i].pos, [p[i].v[0], p[i].v[1]])
        elif mode == 1:
            p[i].motion = control_input(p[i])
            p[i].pos = update_pos(mode, p[i].pos, p[i].motion)

    # 计算逃跑者速度和位置更新
    if not success_flag:
        e.motion = evader_speed(p, e, obstacle, last_pos)
        last_pos = np.array([e.pos[0], e.pos[1]])
        e.pos = update_pos(mode, e.pos, e.motion)
        time_text.set_text("t = %.2fs" % (ii * t))
        # with open("data.txt", 'a') as f:
        #     f.write("%.3f %.3f %.3f\n" % (ii * t, A, dmin))

    return epos, ppos, time_text


ani = animation.FuncAnimation(fig, update, interval=1)
plt.axis('equal')
plt.show()
