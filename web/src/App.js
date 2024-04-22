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

import epipolar from './images/dataset.jpeg';
import structure from './images/gaussian.jpeg';
// import goldberg from './images/octave.jpeg';
import DoG from './images/DoG.png';

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
            0
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

                <div className="flex ttb full-width center">
                  <h2 className="pfTitle half-bottom-margin">
                    A GPU-Accelerated Fast SIFT
                  </h2>

                  <p className="pfContent width-control center-align no-top-margin">
                    {/* Paragraph 1 */}A Scale-Invariant Feature Transform
                    (SIFT) for CUDA enabled devices, with memory optimization
                    and scope for large parallelization.
                  </p>
                  {/* <div className="flex concept">
                    <p className="pfContent pad">Single video</p>
                  </div> */}
                  {/* <object
                    data={right}
                    className="rightArrow"
                    type="image/svg+xml"
                    alt="spiral logo"
                  /> */}

                  {/* <div className="flex concept">
                    <p className="pfContent pad">Structure from Motion</p>
                    </div>
                    <object
                    data={right}
                    className="rightArrow"
                    type="image/svg+xml"
                    alt="spiral logo"
                  /> */}
                  {/* <div className="flex concept">
                    <p className="pfContent pad">Stereo Pair</p>
                  </div> */}
                </div>
              </div>
            </div>

            <div id="pf_con_con">
              <div id="pfs_container" className="medium">
                <div id="introduction_section" className="medium">
                  <p className="pfTitle">Introduction</p>
                  <div className="row">
                    <div>
                      <p>
                        <span className="head"></span>
                      </p>
                      <p>{/* Your introduction content goes here */}</p>
                    </div>
                  </div>
                </div>
                <p className="pfContent width-control">
                  {/* Paragraph 1 */}
                  Our project focuses on accelerating the Scale-Invariant
                  Feature Transform (SIFT) feature extraction process, a
                  widely-used methodology for identifying interest points in
                  images and videos. We aim to reduce the time required for this
                  task by parallelizing various stages of the SIFT algorithm
                  using CUDA-enabled GPUs. This approach substantially boosts
                  the efficiency of video analysis, making it faster and more
                  scalable compared to traditional methods.
                </p>
                <p className="pfContent width-control">
                  The motivation for our project arises from the growing demand
                  for efficient video analysis across industries such as
                  surveillance, entertainment, and healthcare. Our
                  GPU-accelerated solution addresses this need by delivering
                  significant efficiency gains, leading to cost savings and
                  improved productivity for organizations that rely on video
                  analysis for decision-making, monitoring, or research
                  purposes.
                </p>
                {/* Paragraph 3 */}
                <p className="pfContent width-control">
                  For our system, the expected input is a video file, and the
                  output consists of the coordinates or descriptors of the
                  interest points detected in each frame. Specifically, our
                  system will generate a list of interest points along with
                  their corresponding features for each frame. This output
                  facilitates further analysis, object recognition, or tracking
                  tasks, enabling researchers, analysts, and professionals to
                  extract valuable insights from videos more efficiently. By
                  improving the speed and accuracy of video analysis, we aim to
                  enhance the overall efficiency and productivity of video
                  analysis workflows.
                </p>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container" className="medium">
                {/* <p className="pfTitle">DATASET</p> */}
                <div className="row row-left">
                  <div className="img img-left">
                    <img src={epipolar} alt="hello" />
                  </div>
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">1 </span> Capturing Videos for
                        Dataset Creation
                      </span>
                    </p>
                    <p>
                      {' '}
                      {/* Epipolar geometry is fundamental to our project’s goal of
                      generating stereoscopic views from single-camera videos.
                      It involves understanding the geometric relationship
                      between multiple views of the same scene, which is crucial
                      for 3D reconstruction. Using epipolar geometry, we can
                      determine corresponding points between frames, aiding in
                      the estimation of depth and facilitating the creation of a
                      stereoscopic effect. */}
                      We curated our own dataset for the project, focusing on
                      the Coda building. To maintain clarity and sharpness
                      across the frames of the video, we used a gimbal during
                      recording. This ensured minimal blurring and stable
                      footage throughout, enhancing the quality of the video
                      dataset. For accurate camera calibration, we use various
                      checkerboard patterns. These patterns serve as a
                      calibration tool due to their well-defined geometry and
                      known dimensions. The known properties of the checkerboard
                      patterns provides a ground truth reference helping in the
                      precise estimation of camera parameters. To diversify our
                      calibration dataset and avoid symmetrical patterns that
                      could potentially skew the calibration results, we
                      designed custom 3x3 grids.
                    </p>
                  </div>
                </div>
                <div className="row row-right">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">2 </span> Gaussian Filtering for Image Blurring
                      </span>
                    </p>
                    <p>
                      In our project, we utilized a Gaussian filter for image
                      blurring due to its separable nature, which offers
                      computational advantages. Traditional 2D convolution
                      filters require nxm multiplications for each pixel, where
                      n and m are the dimensions of the kernel. In contrast,
                      separable filters can be split into two one-dimensional
                      filters applied sequentially along rows and columns. By
                      taking advantage of this property, we divided the blurring
                      process into horizontal and vertical stages. This method
                      reduces the computational load to just n + m
                      multiplications per pixel, making the Gaussian filter an
                      efficient choice for image blurring while maintaining
                      quality. <br />
                      <br />
                      {/* Smith et al. utilize HDR cameras mounted on a mobile robot
                      for stereo vision-based 3D reconstruction{' '}
                      <span className="lime">[1]</span>. Their method captures
                      textures and spatial features as 2D images and employs an
                      algorithm for depth map visualization.
                      <br />
                      <br />
                      OpenMVG is a library dedicated to Multiple-View Geometry
                      and Structure-from-Motion tasks{' '}
                      <span className="lime">[3]</span>. It facilitates
                      identifying corresponding points between images and
                      manipulating 3D geometry for various computer vision
                      applications. */}
                    </p>
                  </div>
                  <div className="img img-right">
                    <img src={structure} alt="hello" />
                  </div>
                </div>
                <div className="row row-left">
                  <div className="img img-left">
                    <img src={DoG} alt="hello" />
                  </div>
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">3 </span>
                        Difference of Gaussian (DoG)
                      </span>
                    </p>
                    <p>
                      The Difference of Gaussian (DoG) method is employed to
                      effectively identify stable keypoint locations within the
                      scale space of an image. This is achieved by convolving
                      the image with the difference-of-Gaussian function, D(x,
                      y, σ) which is derived from the difference between two
                      neighboring scales separated by a constant multiplicative
                      factor k.One of the primary advantages of using this
                      function is its computational efficiency. As the smoothed
                      images L are already computed for scale space feature
                      description, D can be easily obtained through
                      straightforward image subtraction.
                      <br />
                      <br />
                      Moreover, the DoG function closely approximates the
                      scale-normalized Laplacian of Gaussian. In practice, the
                      initial image undergoes incremental convolution with
                      Gaussians to generate images at different scales, spaced
                      by a constant factor k n scale space. For efficient octave
                      processing, each octave is divided into s intervals,
                      setting k = 2^(1/s) To cover a complete octave during
                      extrema detection, we produce s+3 images in the blurred
                      image stack for each octave. The DoG images, resulting
                      from subtracting adjacent scales, are shown on the left.
                      After processing a full octave, the Gaussian image with
                      double the initial σ value is resampled by selecting every
                      second pixel in each row and column. This resampling
                      maintains accuracy relative to σ while significantly
                      reducing computational overhead.
                    </p>
                  </div>
                </div>
                
                <div className="row row-right">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">4 </span> Resampling
                      </span>
                    </p>
                    <p>
                      In our project, we utilized a Gaussian filter for image
                      blurring due to its separable nature, which offers
                      computational advantages. Traditional 2D convolution
                      filters require nxm multiplications for each pixel, where
                      n and m are the dimensions of the kernel. In contrast,
                      separable filters can be split into two one-dimensional
                      filters applied sequentially along rows and columns. By
                      taking advantage of this property, we divided the blurring
                      process into horizontal and vertical stages. This method
                      reduces the computational load to just n + m
                      multiplications per pixel, making the Gaussian filter an
                      efficient choice for image blurring while maintaining
                      quality. <br />
                      <br />
                      {/* Smith et al. utilize HDR cameras mounted on a mobile robot
                      for stereo vision-based 3D reconstruction{' '}
                      <span className="lime">[1]</span>. Their method captures
                      textures and spatial features as 2D images and employs an
                      algorithm for depth map visualization.
                      <br />
                      <br />
                      OpenMVG is a library dedicated to Multiple-View Geometry
                      and Structure-from-Motion tasks{' '}
                      <span className="lime">[3]</span>. It facilitates
                      identifying corresponding points between images and
                      manipulating 3D geometry for various computer vision
                      applications. */}
                    </p>
                  </div>
                  <div className="img img-right">
                    <img src={structure} alt="hello" />
                  </div>
                </div>
              </div>
            </div>
            {/* <div id="pf_con_con">
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
            </div> */}
            <div id="pf_con_con">
              <div id="pfs_container" className="short">
                <p className="pfTitle">Experiments</p>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        {/* <span className="lime">1 </span>Structure from Motion */}
                      </span>
                    </p>
                    <p>
                      In our experiments, we generated a dataset comprising 1980
                      frames extracted from a 66-second video. To efficiently
                      apply convolution with Gaussian blurring in both vertical
                      and horizontal directions, we processed 128 images
                      simultaneously, arranged in a row. This approach was
                      designed to harness the high DRAM availability of modern
                      GPUs. Specifically, we utilized an NVIDIA H100 80GB HBM3
                      GPU from the Georgia Tech's PACE cluster. With CUDA
                      Version 12.2 and Julia 1.9.2, we leveraged the parallel
                      processing capabilities of our GPU to achieve optimal
                      performance.
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
                      Contributed to the methodology of the project, and making
                      of the website.
                      <br /> <br />
                      Future contributions include generating a GT specific
                      dataset, generating the goldberg polyhedron model for
                      vertices.
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
                      Contributed towards the methodology of the project and
                      implementation of the structure from motion algorithm.{' '}
                      <br /> <br />
                      Future contributions include vertex color generation and
                      color polling based on angle during stereo image-sequence
                      generation.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container" className="short">
                <p className="pfTitle">References</p>
                <div className="row">
                  <div>
                    <p className="cite">
                      <span className="lime cite">1 </span> J. Smith et al.,
                      "Stereo Vision-Based 3D Reconstruction of Indoor Spaces
                      Using HDR Cameras," IEEE Transactions on Robotics, vol.
                      38, no. 4, pp. 567-580, 2023.
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p className="cite">
                      <span className="lime cite">2 </span> A. Jones et al.,
                      "Improved Stereo Image Generation Considering Object
                      Rotation," Applied Sciences, vol. 10, no. 9, p. 3101,
                      2020.
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p className="cite">
                      <span className="lime cite">3 </span> P. Moulon et al.,
                      "OpenMVG: Open Multiple View Geometry," GitHub, 2019.
                      [Online]. Available:{' '}
                      <a href="https://github.com/openMVG" target="_blank">
                        https://github.com/openMVG
                      </a>
                      . [Accessed: Jan. 15, 2022].
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
