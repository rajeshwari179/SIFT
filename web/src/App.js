// import logo from './logo.svg';
import './css/App.css';
import './css/portfolio.css';
import './css/bg.css';
import Portfolio from './components/Portfolio';
import Canvas from './components/Canvas';
import bgShapes1 from './static/Back Shapes_bg.svg';
import bgShapes2 from './static/Back Shapes_bg_2.svg';
import mvl from './static/madewithlove.svg';
import logo from './static/logo.svg';
import right from './static/rightarrow.svg';
import { useState, useEffect, useRef } from 'react';
import {
  rgbToHslHsvHex,
  hexToRgb,
  hsvRgbObjToArr,
  getColourPoints,
} from './js/colorFunctions';

function useInterval(callback, delay) {
  const savedCallback = useRef();

  // Remember the latest callback.
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  // Set up the interval.
  useEffect(() => {
    function tick() {
      savedCallback.current();
    }
    if (delay !== null) {
      let id = setInterval(tick, delay);
      return () => clearInterval(id);
    }
  }, [delay]);
}

function App() {
  var rawCP = {
    colours: [
      '0f1e17',
      '011d26',
      '080b0f',
      '000000',
      '000000',
      '003d1e',
      '0f1e17',
    ],
    // colours: ['000000','FF0000','00FF00','0000FF','FFFF00','00FFFF','FF00FF'],
    radii: [1.3, 1.9, 1.53, 1.01, 1.81, 0.6, 1.11],
    positions: [
      [120.8854, 78.9364],
      [382.0792, 303.7385],
      [108.4034, 492.5733],
      [483.9389, 561.1634],
      [816.4031, 521.2621],
      [832.2423, 592.2427],
      [810.4892, 299.7712],
    ],
    viewport: [841.89, 595.28],
  };
  const [colourPoints, setColourPoints] = useState(getColourPoints(rawCP));
  const [oCP, setOCP] = useState(JSON.parse(JSON.stringify(colourPoints)));
  const [startPerturbation, setStartPerturbation] = useState(false);
  const printColour = (p) => {
    console.log([p.x, p.y]);
  };
  const perturbPos = (first = false) => {
    let cP = JSON.parse(JSON.stringify(colourPoints));
    // console.log(
    //     Math.max(Math.min(oCP[0].x + (Math.random() * 2 - 1) * 1, 1))
    // );
    if (first) {
      for (let i in oCP) {
        oCP[i].x = Math.max(
          Math.min(oCP[i].x + ((Math.random() * 2 - 1) / 2) * 0.5, 1),
          0
        );
        oCP[i].y = Math.max(
          Math.min(oCP[i].y + ((Math.random() * 2 - 1) / 2) * 0.5, 1),
          0
        );
        setOCP([...oCP]);
        // // console.log([cP[i].x, cP[i].y]);
        // cP[i].x = cP[i].x + (Math.random() * 2 - 1) * 0.1;
        // cP[i].y = cP[i].y + (Math.random() * 2 - 1) * 0.1;
        // console.log([cP[i].x, cP[i].y]);
      }
    } else if (startPerturbation) {
      for (let i in oCP) {
        // console.log([cP[i].x, cP[i].y]);
        // if (
        //     oCP[i].x - cP[i].x < 0.0001 ||
        //     oCP[i].y - cP[i].y < 0.0001
        // ) {
        oCP[i].x =
          (((oCP[i].x + ((Math.random() * 2 - 1) / 2) * 0.05) % 1) + 1) % 1;
        oCP[i].y =
          (((oCP[i].y + ((Math.random() * 2 - 1) / 2) * 0.05) % 1) + 1) % 1;
        setOCP([...oCP]);
        // }
        cP[i].x = Math.max(
          Math.min(cP[i].x + 0.005 * Math.random() * (oCP[i].x - cP[i].x), 1),
          0
        );
        cP[i].y = Math.max(
          Math.min(cP[i].y + 0.005 * Math.random() * (oCP[i].y - cP[i].y), 1),
          0
        );
        // console.log(0.05 * Math.random() * (oCP[i].y - cP[i].y));
        // console.log([cP[i].x, cP[i].y]);
      }
    }
    setColourPoints(cP);
  };

  useInterval(perturbPos, 1);

  useEffect(() => {
    perturbPos(true);
    setStartPerturbation(true);
  }, []);

  return (
    <div className="App">
      <div class="backgrounds">
        <Canvas id={'gradientPalette'} canvasPoints={colourPoints} />
        <div className="bg_c_c">
          <div id="bg_Container_1" className="bg_Container">
            <object
              data={bgShapes1}
              id="bgShapes_1"
              type="image/svg+xml"
              alt="bg image"
            />
          </div>
        </div>
        <div className="bg_c_c">
          <div id="bg_Container_2" className="bg_Container">
            <object
              data={bgShapes2}
              id="bgShapes_2"
              type="image/svg+xml"
              alt="another bg image"
            />
          </div>
        </div>
        <object data={logo} id="logo" type="image/svg+xml" alt="spiral logo" />
        <p id="myname">The Stereo Squad</p>
        <object data={mvl} id="mvl" type="image/svg+xml" alt="made with love" />
      </div>
      <div id="pf_con_con">
        <div id="pfs_container">
          <p className="pfTitle">Idea</p>
          <div className="flex ltr">
            <div className='flex concept'>
              <p className="pfContent pad">Single video</p>
            </div>
            <object
              data={right}
              className='rightArrow'
              type="image/svg+xml"
              alt="spiral logo"
            />
            <div className='flex concept'>
              <p className="pfContent pad">Structure from Motion</p>
            </div>
            <object
              data={right}
              className='rightArrow'
              type="image/svg+xml"
              alt="spiral logo"
            />
            <div className='flex concept'>
              <p className="pfContent pad">Stereo Pair</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
