import { Canvas, useThree, useFrame } from "@react-three/fiber";
import { useRef, useLayoutEffect, useEffect } from "react";
import { useTransform, useScroll, useTime } from "framer-motion";
import { degreesToRadians, progress, mix } from "popmotion";

//const color = "#111111";
const color = "#8c8c8c";

const Icosahedron = () => (
    <mesh rotation-x={0.35}>
        <icosahedronGeometry args={[1, 0]} />
        <meshBasicMaterial wireframe color={color} />
    </mesh>
);

const Star = ({ p }: { p: number }) => {
    const ref = useRef<any>(null);

    useLayoutEffect(() => {
        const distance = mix(2, 3.5, Math.random());
        const yAngle = mix(
            degreesToRadians(80),
            degreesToRadians(100),
            Math.random()
        );
        const xAngle = degreesToRadians(360) * p;
        ref.current!.position.setFromSphericalCoords(distance, yAngle, xAngle);
    });

    return (
        <mesh ref={ref}>
            <boxGeometry args={[0.05, 0.05, 0.05]} />
            <meshBasicMaterial wireframe color={color} />
        </mesh>
    );
};

function Scene({ numStars = 100 }) {
    const gl = useThree((state) => state.gl);
    const { scrollYProgress } = useScroll();
    const yAngle = useTransform(
        scrollYProgress,
        [0, 1],
        [0.501, degreesToRadians(180)]
    );
    const distance = useTransform(scrollYProgress, [0, 1], [6, 3]);
    const time = useTime();

    useFrame(({ camera }) => {
        camera.position.setFromSphericalCoords(
            distance.get(),
            yAngle.get(),
            time.get() * 0.00015
        );
        camera.updateProjectionMatrix();
        camera.lookAt(0, 0, 0);
    });

    const stars = [];
    for (let i = 0; i < numStars; i++) {
        stars.push(<Star p={progress(0, numStars, i)} />);
    }

    return (
        <>
            <Icosahedron />
            {stars}
        </>
    );
}

export default function Galaxy() {
    return (
        <Canvas gl={{ antialias: true }} dpr={[1, 2]}>
            <Scene />
        </Canvas>
    );
}
