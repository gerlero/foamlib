import casefoam
import matplotlib.pyplot as plt

# load probe data
p = casefoam.loadProbeData("postProcessing/probes/0/p")
T = casefoam.loadProbeData("postProcessing/probes/0/T")
U = casefoam.loadProbeData("postProcessing/probes/0/U")

# load surface data
surf0_1 = casefoam.loadSurfaceData(
    "postProcessing/freeSurface/0.1/p_freeSurface.raw"
)
surf0_1.sort_values('x', inplace=True)

# load set data
centerLine = casefoam.loadSetData('postProcessing/sample1/0.1/centreLine_T.xy')

p.plot(legend=False)
plt.xlabel('Time $t$')
plt.ylabel('Pressure $p$')
plt.tight_layout()
plt.show()

surf0_1.plot(x='x', y='y', legend=False)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()
plt.show()

centerLine.plot(x='axis', y='value', legend=False)
plt.xlabel('Position $x$')
plt.ylabel('Temperature $T$')
plt.tight_layout()
plt.show()
