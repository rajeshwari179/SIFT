import React, { useState, useRef, useEffect } from 'react';
import renderGradient from '../js/gradientRenderer';

var worker = new window.Worker('./gradientWorker.js');

const Canvas = ({ id, canvasPoints }) => {
    const [unscaledPoints, setUnscaledPoints] = useState(
        JSON.parse(JSON.stringify(canvasPoints))
    );
    const [points, setPoints] = useState([]);
    // const [oPoints] = useState(JSON.parse(JSON.stringify(canvasPoints)));
    const [canvas] = useState(useRef(null));
    const [cSize, setCSize] = useState([]);
    const [windowSize, setWindowSize] = useState({
        width: undefined,
        height: undefined,
    });
    const [firstRender, setFirstRender] = useState(true);

    const draw = (imageData) => {
        var ctx = canvas.current.getContext('2d');
        ctx.putImageData(imageData, 0, 0);
        window.requestAnimationFrame(() => draw(imageData));
    };

    const scaleCP = (Points = unscaledPoints, set = true) => {
        // console.log('scaling to size', Points, set);
        let PointsCopy = JSON.parse(JSON.stringify(Points));
        if (canvas.current.width != canvas.current.clientWidth) {
            canvas.current.width = canvas.current.clientWidth;
        }
        if (canvas.current.height != canvas.current.clientHeight) {
            canvas.current.height = canvas.current.clientHeight;
        }
        for (let i in Points) {
            PointsCopy[i].x = Points[i].x * canvas.current.width;
            PointsCopy[i].y = Points[i].y * canvas.current.height;
        }
        if (set) setPoints([...PointsCopy]);
        return Points;
    };
    const shootPixel = (Points = points) => {
        if (!canvas.current.getContext('webgl2')) {
            console.log('WebGL2 not available, using CPU.');
            var ctx = canvas.current.getContext('2d');
            const imageData = ctx.createImageData(
                canvas.current.width,
                canvas.current.height
            );
            var imDataLength = imageData.data.length;
            // Calling worker
            worker.terminate();
            worker = new window.Worker('./gradientWorker.js');
            worker.postMessage({
                imageData: imageData,
                points: Points,
                canvas: {
                    width: canvas.current.width,
                    height: canvas.current.height,
                },
            });
            worker.onerror = (err) => {
                console.log('error', err);
            };
            worker.onmessage = (e) => {
                if (imDataLength === e.data.imageData.data.length) {
                    window.requestAnimationFrame(() => draw(e.data.imageData));
                }
            };
        } else {
            window.requestAnimationFrame(() =>
                renderGradient(Points, canvas.current)
            );
        }
    };
    useEffect(() => {
        // setPoints(canvasPoints);
        // if (JSON.stringify(canvasPoints) !== JSON.stringify(points)) {
        //     console.log('Rendering new points');
        // }
        scaleCP(canvasPoints);
        shootPixel();
    }, [canvasPoints]);

    useEffect(() => {
        // console.log('inside useEffect');
        if (!firstRender) {
            if (
                canvas.current.clientWidth !== cSize[0] ||
                canvas.current.clientHeight !== cSize[1]
            ) {
                scaleCP(unscaledPoints);
                shootPixel();
            }
        }
    }, [windowSize]);

    useEffect(() => {
        setFirstRender(false);
        if (!canvas.current.getContext('webgl2')) {
            alert(
                'WebGL not available in this browser/platform. Renders may be slower.'
            );
        }
        setCSize([canvas.current.clientWidth, canvas.current.clientHeight]);
        scaleCP(unscaledPoints);
        shootPixel();
        function handleResize() {
            setWindowSize({
                width: window.innerWidth,
                height: window.innerHeight,
            });
        }
        window.addEventListener('resize', handleResize);
    }, []);
    return <canvas id={id} ref={canvas} />;
};

export default Canvas;
