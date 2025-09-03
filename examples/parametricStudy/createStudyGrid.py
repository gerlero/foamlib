from pathlib import Path

from foamlib.preprocessing.grid_parameter_sweep import CaseParameter, GridParameter
from foamlib.preprocessing.of_dict import FoamDictInstruction
from foamlib.preprocessing.parameter_study import grid_generator
from foamlib.preprocessing.system import simulationParameters

# damBreak
root = Path(__file__).parent
template_case = root / "damBreak"


def grid_parameters(scale) -> list[int]:
    return [
        int(23 * scale),
        int(8 * scale),
        int(19 * scale),
        int(42 * scale),
        int(4 * scale),
    ]


grid = GridParameter(
    parameter_name="grid",
    # generate 5 instructions in system/simulationParameters with the key1..5
    # This is simulationParameters is identical to the following:
    # FoamDictInstruction(
    #     file_name=Path("system/simulationParameters"),
    #     keys=[f"res{i}"],
    # )
    modify_dict=[simulationParameters(keys=[f"res{i}"]) for i in range(1, 6)],
    parameters=[
        CaseParameter(name="coarse", values=grid_parameters(1)),
        CaseParameter(name="mid", values=grid_parameters(2)),
        CaseParameter(name="fine", values=grid_parameters(4)),
    ],
)

init_height = GridParameter(
    parameter_name="initHeight",
    modify_dict=[simulationParameters(keys=["initHeight"])],
    parameters=[
        CaseParameter(name="height_02", values=[0.2]),
        CaseParameter(name="height_03", values=[0.3]),
        CaseParameter(name="height_04", values=[0.4]),
    ],
)

study = grid_generator(
    parameters=[grid, init_height],
    template_case=template_case,
    output_folder=root / "Cases",
)

study.create_study(study_base_folder=root)
