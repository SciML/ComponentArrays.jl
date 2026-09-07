using PrecompileTools: @compile_workload

@compile_workload begin
    component = ComponentArray(position = [1.0, 2.0], velocity = [3.0, 4.0])
    component.position
    component[:velocity]
    component[1:2]
    getdata(component)
    getaxes(component)
    axes(component)
end
