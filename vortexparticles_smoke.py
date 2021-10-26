
import math

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

eps = 0.00001
dt = 0.02

numberOfParticles = 20000

positions = ti.Vector.field(2, ti.f32, shape=numberOfParticles)
new_pos = ti.Vector.field(2, ti.f32, shape=numberOfParticles)

vorticities = ti.Vector.field(1, ti.f32, shape = (numberOfParticles))


#存储温度场和浮力场的网格的分辨率
t_resolutionX = 200
t_resolutionY = t_resolutionX


gridSpacing = 1.0 / t_resolutionX

#调节浮力的参数
alpha = 0.0001
beta  = 0.00001

#存储温度场
TGrid = ti.Vector.field(1, ti.f32, shape = (t_resolutionX, t_resolutionY))
#存储浮力场
BGrid = ti.Vector.field(1, ti.f32, shape = (t_resolutionX, t_resolutionY))
#网格节点的位置
grid_positions = ti.Vector.field(2, ti.f32, shape = (t_resolutionX, t_resolutionY))
#存放浮力场的旋度
buoyancy_curl = ti.Vector.field(1, ti.f32, shape = (t_resolutionX, t_resolutionY))



@ti.kernel
def emmitParticles():
    for i in range(numberOfParticles):
        positions[i][0] = ti.random()/5 + 0.4
        positions[i][1] = ti.random()/5 + 0.2




#初始化网格节点的位置
@ti.kernel
def init_gpositions():
    for j in range(t_resolutionX):
        for i in range(t_resolutionY):
            grid_positions[i, j] = ti.Vector([i * gridSpacing, j * gridSpacing])






#初始化温度场
@ti.kernel
def init_TGrid():

    heatS_center = ti.Vector([0.5, 0.5])
    heatS_r = 0.15
    heatS_r2 = heatS_r *heatS_r
    for j in range(t_resolutionY):
        for i in range(t_resolutionX):
            p1 = i * gridSpacing
            p2 = j * gridSpacing

            p = ti.Vector([p1, p2])
            p_r = p.norm()
            p_r2 = p_r * p_r
            if((heatS_center - p).norm() <heatS_r):
                TGrid[i, j][0] = (grid_positions[i, j][1]) * ti.random() * p_r
            if (p2 < 0.28):
                TGrid[i, j][0] = 0

            if(p2 > 0.5):
                TGrid[i, j][0] = ti.random()



#计算浮力,我们假设高度越高温度越低
#发射源处的温度特别处理
@ti.kernel
def calculate_buoyncy():
    for j in range(t_resolutionY):
        for i in range(t_resolutionX):
            BGrid[i,j][0] = beta * ((TGrid[i,j][0]) * ti.Vector([0, 1]))[1]



#求解浮力场的旋度场
#由于假设浮力只有竖直方向的分量，因此这里的旋度运算省区y方向的计算
@ti.func
def solveBuoyancyCurl():
    for j in range(t_resolutionY):
        for i in range(t_resolutionX):
            if (i+1 >= t_resolutionX):
               buoyancy_curl[i,j][0] = (BGrid[i,j][0] - BGrid[i-1, j][0]) / gridSpacing
               continue
            if (i-1 < 0):
               buoyancy_curl[i,j][0] = (BGrid[i+1,j][0] - BGrid[i,j][0]) / gridSpacing
               continue
            buoyancy_curl[i,j][0] =  (BGrid[i+1,j][0] - BGrid[i-1, j][0]) / gridSpacing



#双线性插值
#S1  S2

#S4  S3

#求解由于浮力引起的涡度
@ti.func
def buoyancyAdvection():
    for i in range(numberOfParticles):

        gridS = gridSpacing * gridSpacing
        index = positions[i] // gridSpacing
        origin = index
        leftTop = origin + ti.Vector([0,1])
        rightDown = origin + ti.Vector([1,0])
        rightTop = origin + ti.Vector([1,1])

        #粒子在单元网格中的相对坐标
        posInGrid = positions[i] - grid_positions[index[0], index[1]]
        S1 = posInGrid[0] * (gridSpacing - posInGrid[1])
        S2 = (gridSpacing - posInGrid[0]) * (gridSpacing - posInGrid[1])
        S3 = (gridSpacing - posInGrid[0]) * posInGrid[1]
        S4 = posInGrid[0] * posInGrid[1]
        result = (S1 * buoyancy_curl[rightDown[0],rightDown[1]][0] + \
                 S2 * buoyancy_curl[origin[0],origin[1]][0] + \
                 S3 * buoyancy_curl[leftTop[0],leftTop[1]][0] + \
                 S4 * buoyancy_curl[rightTop[0],rightTop[1]][0]) / gridS

        #这里将来换成二阶的龙格库塔积分
        tempVor = result * dt
        vorticities[i][0] += tempVor




@ti.func
def compute_u_single(p, i):
    r2 = (p - positions[i]).norm()**2
    uv = ti.Vector([positions[i].y - p.y, p.x - positions[i].x])
    return vorticities[i][0] * uv / (r2 * math.pi) * 0.5 * (1.0 - ti.exp(-r2 / eps**2))

@ti.func
def compute_u_single_other(p, i):
    r = (p - positions[i]).norm()
    r2 = r**2
    uv = ti.Vector([positions[i].y - p.y, p.x - positions[i].x])
    return vorticities[i][0] * uv / (math.pi * (r2 + eps*ti.exp(-r/eps))) * 0.5 * (1.0 - ti.exp(-r / eps))



@ti.func
def integrate_vortex():
    for i in range(numberOfParticles):
        v = ti.Vector([0.0, 0.0])
        for j in range(numberOfParticles):
            if i != j:
                #v += compute_u_single(positions[i], j)
                v+=compute_u_single_other(positions[i], j)
        new_pos[i] = positions[i] + dt * v

    for i in range(numberOfParticles):
        positions[i] = new_pos[i]


@ti.kernel
def simulation():
    solveBuoyancyCurl()
    buoyancyAdvection()
    integrate_vortex()


def main():
    init_gpositions()
    emmitParticles()
    init_TGrid()
    calculate_buoyncy()


gui = ti.GUI("Vortex smoke", (512, 512), background_color=0x0000)
main()
while(True):
    simulation()
    gui.circles(positions.to_numpy(),
                radius=1,
                color=0xffffff)

    gui.show()
