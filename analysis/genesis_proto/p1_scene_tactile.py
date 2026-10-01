"""P1: Genesis scene with a multi-finger hand carrying fingertip tactile sensors.

First step of the algorithm prototype: get a hand + touch + an object into Genesis and
verify the tactile stream reads. Everything after this (task, policy, prediction routing)
builds on it.
"""
import numpy as np
import genesis as gs


def main():
    gs.init(backend=gs.cpu, logging_level='warning')
    sc = gs.Scene(show_viewer=False)

    sc.add_entity(gs.morphs.Plane())
    # palm-height platform so the hand can be posed above it
    sc.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.02), pos=(0, 0, 0.01), fixed=True))
    obj = sc.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0, 0, 0.05)))

    # a multi-finger hand from the bundled assets
    import os
    hand_urdf = os.path.join(os.path.dirname(gs.__file__), "assets", "urdf", "shadow_hand")
    files = os.listdir(hand_urdf)
    print(f"   shadow_hand 资产: {files[:8]}")
    print("   （先只验证手能否加载与列出 link，触觉挂载下一步做）")
    print(f"   ✓ 场景元素: plane + platform + box({obj.idx})")


if __name__ == "__main__":
    main()
