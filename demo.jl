using CUDA
using GLMakie
using StaticArrays

const SVec3 = SVector{3, Float32}

targetFps = 60
numSubsteps = 30
timeStep = 1.0 / 60.0
gravity = SVec3(0, -10, 0)
paused = false 
hidden = false 
frameNr = 0

clothNumX = 500
clothNumY = 500
clothY = 2.2
clothSpacing = 0.01
sphereCenter = SVec3(0, 1.5, 0)
sphereRadius = 0.5


#function init(yOffset, numX, numY, spacing, sphereCenter, sphereRadiuss)

numX = clothNumX
numY = clothNumY
spacing = clothSpacing
yOffset = clothY

dragParticleNr = -1
dragDepth = 0.0
dragInvMass = 0.0

numX += numX % 2
numY += numY % 2 

numParticles = (numX + 1) * (numY + 1)
hostPos = zeros(SVec3, numX + 1, numY + 1)
hostNormals = zeros(SVec3, numX + 1, numY + 1)
hostInvMass = zeros(numParticles)

faces = zeros(Int, numX * numY * 2, 3)
k = 1

for I in CartesianIndices(hostPos)
    id = LinearIndices(hostPos)[I]

    hostPos[I] = SVec3(
                (-numX * 0.5 + I[1])*spacing, 
                yOffset, 
                (-numY * 0.5 + I[2])*spacing
            )

    if I[1] <= numX && I[2] <= numY
        tI = Tuple(I)
        faces[k, :] .= (id, id + 1, id + numY + 1)
        faces[k+1, :] .= (id + 1, id + numY + 2, id + numY + 1)
        k += 2
    end
end

pos = cu(hostPos)
prevPos = cu(hostPos)
restPost = cu(hostPos)
invMass = cu(hostInvMass)
corr = CUDA.zeros(SVec3, numX+1, numY+1)
vel = CUDA.zeros(SVec3, numX+1, numY+1)
normals = cu(hostNormals)

# constraints




begin 
    fig = Figure() 
    ax = Axis3(fig[1,1])
    mesh!(vec(hostPos), faces)
    fig 
end




