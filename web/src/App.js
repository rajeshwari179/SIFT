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
import done from './static/done.svg';
import pending from './static/pending.svg';
import main from './static/main img.svg';
import { MathJaxContext, MathJax } from 'better-react-mathjax';
import { useState, useEffect, useRef } from 'react';
import {
  rgbToHslHsvHex,
  hexToRgb,
  hsvRgbObjToArr,
  getColourPoints,
} from './js/colorFunctions';

import epipolar from './images/epipolar geometry.png';
import structure from './images/structure from motion.png';
import goldberg from './images/Goldberg.png';

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
    <MathJaxContext>
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
          <object
            data={logo}
            id="logo"
            type="image/svg+xml"
            alt="spiral logo"
          />
          <p id="myname">The Stereo Squad</p>
          <object
            data={mvl}
            id="mvl"
            type="image/svg+xml"
            alt="made with love"
          />
        </div>
        <div className="main-box">
          <div className="main-content">
            <div id="pf_con_con">
              <div id="pfs_container" className="medium">
                <p className="pfTitle">Idea</p>
                <div className="main-img-box">
                  <object
                    data={main}
                    className="main"
                    type="image/svg+xml"
                    alt="spiral logo"
                  />
                </div>

                <div className="flex ltr full-width center">
                  <div className="flex concept">
                    <p className="pfContent pad">Single video</p>
                  </div>
                  <object
                    data={right}
                    className="rightArrow"
                    type="image/svg+xml"
                    alt="spiral logo"
                  />
                  {/* <div className="flex concept">
                    <p className="pfContent pad">Structure from Motion</p>
                    </div>
                    <object
                    data={right}
                    className="rightArrow"
                    type="image/svg+xml"
                    alt="spiral logo"
                  /> */}
                  <div className="flex concept">
                    <p className="pfContent pad">Stereo Pair</p>
                  </div>
                </div>
                <p className="pfContent width-control">
                  Generate a stereo pair of a given video from the video? Videos
                  are essentially frames of images with camera at different 3D
                  positions. Reconstructing a 3D model of the world would allow
                  for the generation the stereo pair of the input video.
                  Although it won't be perfect, we predict that we can use a
                  GenAI to filter and smoothen the output, but is beyond the
                  scope of this project.
                </p>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container" className="medium">
                <p className="pfTitle">Math + Engineering</p>
                <div className="row row-left">
                  <div className="img img-left">
                    <img src={epipolar} alt="hello" />
                  </div>
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">1 </span>Epipolar Geometry
                      </span>
                    </p>
                    <p>
                      {' '}
                      Epipolar geometry is fundamental to our project’s goal of
                      generating stereoscopic views from single-camera videos.
                      It involves understanding the geometric relationship
                      between multiple views of the same scene, which is crucial
                      for 3D reconstruction. Using epipolar geometry, we can
                      determine corresponding points between frames, aiding in
                      the estimation of depth and facilitating the creation of a
                      stereoscopic effect.
                    </p>
                  </div>
                </div>
                <div className="row row-right">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">2 </span>Structure From Motion
                      </span>
                    </p>
                    <p>
                      SfM is a critical component of our approach to 3D
                      reconstruction from motion video. It allows us to extract
                      three-dimensional structure from two-dimensional image
                      sequences, which is essential for our non-learning-based
                      3D reconstruction method. Utilizing SfM, we can
                      reconstruct scene geometry and camera poses, enabling us
                      to generate a stereoscopic view without the need for a
                      stereo-camera setup
                    </p>
                  </div>
                  <div className="img img-right">
                    <img src={structure} alt="hello" />
                  </div>
                </div>
                <div className="row row-left">
                  <div className="img img-left">
                    <img src={goldberg} alt="hello" />
                  </div>
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">3 </span>
                        Vertex Colors and Angle-Based Binning
                      </span>
                    </p>
                    <p>
                      Our approach for coloring vertices takes into account the
                      viewing angle. Instead of a single fixed color, we divide
                      the full 360-degree range around each vertex into
                      equal-sized bins. These bins form a Goldberg polynomial
                      shape. For each vertex, we assign different colors to
                      these bins. During stereo image generation, our algorithm
                      looks up the color based on the viewing angle bin. This
                      technique ensures that our 3D scenes have realistic color
                      variations, enhancing the overall visual experience.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container">
                <p className="pfTitle">Progress...</p>
                <div className="flex column">
                  <div className="flex">
                    <object
                      data={done}
                      className="rightArrow status"
                      type="image/svg+xml"
                      alt="spiral logo"
                    />
                    <p className="no-wrap">Ideation</p>
                  </div>
                  <div className="flex">
                    <object
                      data={done}
                      className="rightArrow status"
                      type="image/svg+xml"
                      alt="spiral logo"
                    />
                    <p className="no-wrap">Website Creation</p>
                  </div>
                  <div className="flex">
                    <object
                      data={done}
                      className="rightArrow status"
                      type="image/svg+xml"
                      alt="spiral logo"
                    />
                    <p className="no-wrap">Implement Structure from Motion</p>
                  </div>
                  <div className="flex">
                    <object
                      data={pending}
                      className="rightArrow status"
                      type="image/svg+xml"
                      alt="spiral logo"
                    />
                    <p className="left">
                      Generate GT specific dataset, using a gimball and
                      checkerboard patterns, at coda
                    </p>
                  </div>
                  <div className="flex">
                    <object
                      data={pending}
                      className="rightArrow status"
                      type="image/svg+xml"
                      alt="spiral logo"
                    />
                    <p className="left">
                      Goldberg Polyhedron Model for Vertices
                    </p>
                  </div>
                  <div className="flex">
                    <object
                      data={pending}
                      className="rightArrow status"
                      type="image/svg+xml"
                      alt="spiral logo"
                    />
                    <p className="left">Vertex Color Generation</p>
                  </div>
                  <div className="flex">
                    <object
                      data={pending}
                      className="rightArrow status"
                      type="image/svg+xml"
                      alt="spiral logo"
                    />
                    <p className="left">Generate Stereo Video</p>
                  </div>
                </div>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container" className="short">
                <p className="pfTitle">Experiments</p>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">1 </span>Structure from Motion
                      </span>
                    </p>
                    <p>
                      We tried to create a 3D plot of points of a video using 10
                      frames. The points were extracted using the SIFT
                      algorithm. The 3D plot was created using an SfM algorithm
                      that we found online. The plot was not very accurate, but
                      it was a good start. The next step is to improve the
                      accuracy of the 3D plot, change the algorithm so that it
                      certainly uses the checkerboard pattern we placed in
                      world.
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">2 </span> Vertex Colors
                      </span>
                    </p>
                    <p>
                      We first postulated that a vertex will have only one
                      color. We then changed our approach to have a vertex have
                      multiple colors based on the angle of view. We are
                      currently working on implementing this approach. This will
                      allow us to have a more realistic 3D model and
                      characterize specular effects better.
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">3 </span> Interest Points
                      </span>
                    </p>
                    <p>
                      Although our approach primarily relies on interest points
                      being necessarily vertices, we are also considering the
                      possibility of using lines and their angles as another
                      "feature"-set. This will probably allow us to generate
                      surfaces using triangular meshes and will be a more robust
                      approach.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container" className="short">
                <p className="pfTitle">Contributions</p>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">1 </span>Kishore Shenoy
                      </span>
                    </p>
                    <p>
                      Contributed to the methodology of the project, and making of the website.
                      <br/> <br/>
                      Future contributions include generating a GT specific dataset, generating the goldberg polyhedron model for vertices.
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">2 </span> Rajeshwari Devaramani
                      </span>
                    </p>
                    <p>
                      Contributed towards the methodology of the project and implementation of the structure from motion algorithm. <br/> <br/>

                      Future contributions include vertex color generation and color polling based on angle during stereo image-sequence generation.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </MathJaxContext>
  );
}

export default App;
