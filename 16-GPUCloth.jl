# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This implementation is a based on:
# www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics

using LinearAlgebra
using CUDA, KernelAbstractions, Atomix
using GLMakie, GeometryBasics
using StaticArrays

const SVec3 = SVector{3, Float32}

function init_state(;
        numSubsteps = 30,
        timeStep = 1.0f0 / 60.0f0,
        gravity = SVec3(0, 0, -10),
        sphereCenter = SVec3(0, 0, 1.5),
        sphereRadius = 0.5f0,
        numX = 500,
        numY = 500,
        spacing = 0.01f0,
        zOffset = 2.2f0
    )
    # targetFps = 60
    # paused = false 
    # hidden = false 
    # frameNr = 0

    # dragParticleNr = -1
    # dragDepth = 0.0
    # dragInvMass = 0.0

    numX += numX % 2
    numY += numY % 2 

    numParticles = (numX + 1) * (numY + 1)
    hostPos = zeros(SVec3, numX + 1, numY + 1)
    # hostNormals = zeros(SVec3, 2 * numX * numY)
    hostInvMass = ones(Float32, numParticles)
    hostTriIds = Vector{GLTriangleFace}(undef, 2 * numX * numY)


    k = 1
    for I in CartesianIndices(hostPos)
        id = LinearIndices(hostPos)[I]

        hostPos[I] = SVec3(
                    (-numX * 0.5 + I[1] - 1)*spacing, 
                    (-numY * 0.5 + I[2] - 1)*spacing,
                    zOffset
                )

        if I[1] <= numX && I[2] <= numY
            tI = Tuple(I)
            hostTriIds[k] = GLTriangleFace(id, id + 1, id + numY + 1)
            hostTriIds[k+1] = GLTriangleFace(id + 1, id + numY + 2, id + numY + 1)
            k += 2
        end
    end

    pos = cu(hostPos)
    prevPos = cu(hostPos)
    restPos = cu(hostPos)
    invMass = cu(hostInvMass)
    corr = CUDA.zeros(SVec3, numX+1, numY+1)
    vel = CUDA.zeros(SVec3, numX+1, numY+1)
    # normals = cu(hostNormals)
    triIds = cu(hostTriIds)

    # constraints
    passSizes = (
        (numX + 1) * floor(Int, numY / 2),
        (numX + 1) * floor(Int, numY / 2),
        floor(Int, numX / 2) * (numY + 1),
        floor(Int, numX / 2) * (numY + 1),
        2 * numX * numY + (numX + 1) * (numY - 1) + (numY + 1) * (numX - 1)
    )


    passIndependent = (true, true, true, true, false)

    numDistConstraints = sum(passSizes)

    hostDistConstIds = Array{NTuple{2, Int32}}(undef, numDistConstraints)

    # stretch constraints
    begin
        k = 1
        for passNr in 1:2
            for i in passNr:2:size(pos,1)-1, j in 1:size(pos, 2)
                hostDistConstIds[k] = (LinearIndices(pos)[i,j], LinearIndices(pos)[i+1,j])
                k += 1
            end
        end

        for passNr in 1:2
            for i in 1:size(pos, 1), j in passNr:2:size(pos,2)-1
                hostDistConstIds[k] = (LinearIndices(pos)[i,j], LinearIndices(pos)[i,j+1])
                k += 1
            end
        end

        # shear constraints 

        for i in 1:size(pos,1)-1, j in 1:size(pos,2)-1
            hostDistConstIds[k] = (LinearIndices(pos)[i,j], LinearIndices(pos)[i+1,j+1])
            k += 1
            hostDistConstIds[k] = (LinearIndices(pos)[i+1,j], LinearIndices(pos)[i,j+1])
            k += 1
        end

        # bending constraints 

        for i in 1:size(pos,1)-2, j in 1:size(pos,2)
            hostDistConstIds[k] = (LinearIndices(pos)[i,j], LinearIndices(pos)[i+2,j])
            k += 1
        end

        for i in 1:size(pos,1), j in 1:size(pos,2)-2
            hostDistConstIds[k] = (LinearIndices(pos)[i,j], LinearIndices(pos)[i,j+2])
            k += 1
        end
    end

    distConstIds = cu(hostDistConstIds)

    @kernel function computeRestLengths(pos, constIds, restLengths)
        cNr = @index(Global)
        p0 = pos[constIds[cNr][1]]
        p1 = pos[constIds[cNr][2]]
        restLengths[cNr] = norm(p0 - p1)
    end

    constRestLengths = CUDA.zeros(numDistConstraints)

    backend = get_backend(pos)
    computeRestLengths(backend, 64)(pos, distConstIds, constRestLengths, ndrange = numDistConstraints)
    KernelAbstractions.synchronize(backend)


    state = (;
        pos, 
        vel,
        triIds,
        # normals,
        prevPos,
        invMass,
        corr,
        passSizes, 
        passIndependent,
        distConstIds,
        constRestLengths, 
        restPos,
        hostPos,
        hostTriIds,
        # hostNormals
    )

    params = (;
        timeStep,    
        numSubsteps,
        gravity, 
        sphereCenter = sphereCenter + SVec3(0,0,0),
        sphereRadius,
        jacobiScale = 0.2f0
    )

    return (;state, params)
end

# helper function to support atomic add of static vectors
function vecAtomicAdd(x, i, y::SVec3)
    x_ = reshape(reinterpret(Float32, x), 3, length(x))
    Atomix.@atomic x_[1,i] += y[1]
    Atomix.@atomic x_[2,i] += y[2]
    Atomix.@atomic x_[3,i] += y[3]
end

# compute normals 
# @kernel function addNormals(@Const(pos), @Const(triIds), normals)
#     triNr = @index(Global)

#     id0, id1, id2 = convert.(UInt32, triIds[triNr])
#     normal = cross(pos[id1] - pos[id0], pos[id2] - pos[id0])
#     vecAtomicAdd(normals, id0, normal)
#     vecAtomicAdd(normals, id1, normal)
#     vecAtomicAdd(normals, id2, normal)
# end

# @kernel function normalizeNormals(normals)
#     normalId = @index(Global)
#     normals[normalId] = normalize(normals[normalId]) 
# end


@kernel function integrate(dt, gravity, invMass, prevPos, pos, vel, sphereCenter, sphereRadius)
    pNr = @index(Global)

    prevPos[pNr] = pos[pNr]
    if invMass[pNr] != 0.0f0
        vel[pNr] += gravity * dt 
        pos[pNr] += vel[pNr] * dt 

        thickness = 0.001f0
        friction = 0.01f0

        d = norm(pos[pNr] - sphereCenter)
        if d < (sphereRadius + thickness)
            p = pos[pNr] * (1.0f0 - friction) + prevPos[pNr] * friction 
            r = p - sphereCenter
            d = norm(r)
            pos[pNr] = sphereCenter + r * ((sphereRadius + thickness) / d)
        end

        p = pos[pNr]
        if p[3] < thickness
            p = pos[pNr] * (1.0f0 - friction) + prevPos[pNr] * friction 
            pos[pNr] = SVec3(p[1], p[2], thickness)
        end
    end
end

@kernel function updateVel(dt, prevPos, pos, vel)
    pNr = @index(Global)
    vel[pNr] = (pos[pNr] - prevPos[pNr]) / dt
end

@kernel function solveDistanceConstraints(useJaccobi, firstConstraint, invMass, pos, corr, constIds, restLengths)
    cNr = firstConstraint + @index(Global)
    id0, id1 = constIds[cNr]
    w0 = invMass[id0]
    w1 = invMass[id1]

    w = w0 + w1 
    if w != 0.0 
        p0 = pos[id0]
        p1 = pos[id1]

        d = p1 - p0 
        l = norm(d)
        n = d ./ l
        l0 = restLengths[cNr]
        dP = n * (l - l0) / w 
        if useJaccobi == true
            vecAtomicAdd(corr, id0, w0 * dP)
            vecAtomicAdd(corr, id1, -w1 * dP)
        else
            vecAtomicAdd(pos, id0, w0 * dP)
            vecAtomicAdd(pos, id1, -w1 * dP)
        end
    end
end

@kernel function addCorrections(pos, corr, scale)
    pNr = @index(Global)
    pos[pNr] += corr[pNr] * scale 
end

function simulate!(s, p, backend = CUDABackend(), nworkgroups = 256)  # state, params
    (;pos, vel, prevPos, hostPos, invMass, corr, distConstIds, constRestLengths, passSizes, passIndependent) = s
    # (;normals, triIds, hostNormals) = s
    (;timeStep, numSubsteps, gravity, sphereCenter, sphereRadius, jacobiScale) = p
    
    dt = timeStep / numSubsteps
    for step in 1:numSubsteps
        integrate(backend, nworkgroups)(dt, gravity, invMass, prevPos, pos, vel, sphereCenter, sphereRadius, ndrange = length(pos))

        firstConstraint = 0
        for passNr in 1:5
            numConstraints = passSizes[passNr]

            if passIndependent[passNr]
                # use Gauss-Seidel 
                solveDistanceConstraints(backend, nworkgroups)(false, firstConstraint, invMass, pos, corr, distConstIds, constRestLengths, ndrange = numConstraints)
            else 
                # use Jaccobi 
                fill!(corr, zero(SVec3))
                solveDistanceConstraints(backend, nworkgroups)(true, firstConstraint, invMass, pos, corr, distConstIds, constRestLengths, ndrange = numConstraints)
                addCorrections(backend, nworkgroups)(pos, corr, jacobiScale, ndrange = length(pos))
            end

            firstConstraint += numConstraints
        end
        updateVel(backend, nworkgroups)(dt, prevPos, pos, vel, ndrange = length(vel))

        #TODO: add constraint solvers
    end
    
    # update normals 
    # fill!(normals, zero(SVec3))
    # addNormals(backend, nworkgroups)(pos, triIds, normals, ndrange = length(triIds))
    # normalizeNormals(backend, nworkgroups)(normals, ndrange = length(normals))
    # KernelAbstractions.synchronize(backend)

    copy!(hostPos, pos)
    # copy!(hostNormals, normals)
end


function reset!(s)
    fill!(s.vel, zero(SVec3))
    copy!(s.pos, s.restPos)
    copy!(s.hostPos, s.pos)
end




# visualisation

function init_plot(state, params)
    
    # prepare data pipeline
    stateObs = Observable(state)
    msh = @lift GeometryBasics.Mesh(reinterpret(Point3f, vec($stateObs.hostPos)), state.hostTriIds)
    
    # lights
    lights = [
        DirectionalLight(RGBf(0, 0, 0.7), Vec3f(-4, -4, 0)),
        DirectionalLight(RGBf(0.7, 0.2, 0), Vec3f(-4, 4, -1)),
        DirectionalLight(RGBf(0.7, 0.7, 0.7), Vec3f(4, -4, -1)),
        AmbientLight(RGBf(0.3, 0.2, 0.2))
    ]

    set_theme!(theme_dark())
    fig = Figure(size = (800, 800)) 
    ax = LScene(fig[1,1], scenekw = (;lights, camera = cam3d!,), show_axis = false)

    mesh!(msh, transparency = true, color = (:orange, 0.4), backlight = 1)
    meshscatter!([Point3f(params.sphereCenter)], markersize = params.sphereRadius, color = :red)
    cam = cameracontrols(ax.scene)
    cam.lookat[] = params.sphereCenter - Vec(0,0,1)
    rotate_cam!(ax.scene, cam, (0.0, -pi/5, 0.0))
    update_cam!(ax.scene, cam)

    display(fig) 

    return fig, stateObs
end

function update_plot!(stateObs, state, params)
    stateObs[] = state
end



# reset!(state)
# update_plot!(posObs, state, params)
# for i in 1:200 
#     simulate!(state, params)
#     update_plot!(posObs, state, params)
#     sleep(0.001)
# end



# init 
state, params = init_state()


begin 
    reset!(state)
    fig, stateObs = init_plot(state, params)
    ax = content(fig[1,1])

    Makie.record(fig, "16-GPUCloth.mp4", 1:200; update = false, visible=true) do i 
        simulate!(state, params)
        update_plot!(stateObs, state, params)

        cam = cameracontrols(ax.scene)
        if i == 1  # I don't know how to change initial camera with LScene...

            cam.lookat[] = params.sphereCenter - Vec(0,0,1)
            translate_cam!(ax.scene, cam, Vec3f(0.0, 0.0, -2))
            zoom!(ax.scene, cam, 0.75)
            update_cam!(ax.scene, cam)
        end
    end
end



# @time simulate!(state, params, CUDABackend(), 128)
# 0.034735 seconds (10.44 k allocations: 409.734 KiB)
