import {
  Scene,
  CubeCamera,
  Object3D,
  Vector3,
  BoxBufferGeometry,
  ShaderMaterial,
  UniformsUtils,
  BackSide,
  Mesh,
  UniformsLib,
  NoBlending
} from "three";

import * as THREE from "three";

/**
 * @author Prashant Sharma / spidersharma03
 * @author Ben Houston / bhouston, https://clara.io
 *
 * This class takes the cube lods(corresponding to different roughness values), and creates a single cubeUV
 * Texture. The format for a given roughness set of faces is simply::
 * +X+Y+Z
 * -X-Y-Z
 * For every roughness a mip map chain is also saved, which is essential to remove the texture artifacts due to
 * minification.
 * Right now for every face a PlaneMesh is drawn, which leads to a lot of geometry draw calls, but can be replaced
 * later by drawing a single buffer and by sending the appropriate faceIndex via vertex attributes.
 * The arrangement of the faces is fixed, as assuming this arrangement, the sampling function has been written.
 */

const PMREMCubeUVPacker = (function() {
  const camera = new THREE.OrthographicCamera();
  const scene = new THREE.Scene();
  const shader = getShader();

  const PMREMCubeUVPacker = function(cubeTextureLods) {
    this.cubeLods = cubeTextureLods;
    let size = cubeTextureLods[0].width * 4;

    const sourceTexture = cubeTextureLods[0].texture;
    const params = {
      format: sourceTexture.format,
      magFilter: sourceTexture.magFilter,
      minFilter: sourceTexture.minFilter,
      type: sourceTexture.type,
      generateMipmaps: sourceTexture.generateMipmaps,
      anisotropy: sourceTexture.anisotropy,
      encoding: sourceTexture.encoding === THREE.RGBEEncoding ? THREE.RGBM16Encoding : sourceTexture.encoding
    };

    if (params.encoding === THREE.RGBM16Encoding) {
      params.magFilter = THREE.LinearFilter;
      params.minFilter = THREE.LinearFilter;
    }

    this.CubeUVRenderTarget = new THREE.WebGLRenderTarget(size, size, params);
    this.CubeUVRenderTarget.texture.name = "PMREMCubeUVPacker.cubeUv";
    this.CubeUVRenderTarget.texture.mapping = THREE.CubeUVReflectionMapping;

    this.objects = [];

    const geometry = new THREE.PlaneBufferGeometry(1, 1);

    const faceOffsets = [];
    faceOffsets.push(new THREE.Vector2(0, 0));
    faceOffsets.push(new THREE.Vector2(1, 0));
    faceOffsets.push(new THREE.Vector2(2, 0));
    faceOffsets.push(new THREE.Vector2(0, 1));
    faceOffsets.push(new THREE.Vector2(1, 1));
    faceOffsets.push(new THREE.Vector2(2, 1));

    const textureResolution = size;
    size = cubeTextureLods[0].width;

    let offset2 = 0;
    let c = 4.0;
    this.numLods = Math.log(cubeTextureLods[0].width) / Math.log(2) - 2; // IE11 doesn't support Math.log2
    for (let i = 0; i < this.numLods; i++) {
      const offset1 = (textureResolution - textureResolution / c) * 0.5;
      if (size > 16) c *= 2;
      const nMips = size > 16 ? 6 : 1;
      let mipOffsetX = 0;
      let mipOffsetY = 0;
      let mipSize = size;

      for (let j = 0; j < nMips; j++) {
        // Mip Maps
        for (let k = 0; k < 6; k++) {
          // 6 Cube Faces
          const material = shader.clone();
          material.uniforms["envMap"].value = this.cubeLods[i].texture;
          material.envMap = this.cubeLods[i].texture;
          material.uniforms["faceIndex"].value = k;
          material.uniforms["mapSize"].value = mipSize;

          const planeMesh = new THREE.Mesh(geometry, material);
          planeMesh.position.x = faceOffsets[k].x * mipSize - offset1 + mipOffsetX;
          planeMesh.position.y = faceOffsets[k].y * mipSize - offset1 + offset2 + mipOffsetY;
          planeMesh.material.side = THREE.BackSide;
          planeMesh.scale.setScalar(mipSize);
          this.objects.push(planeMesh);
        }
        mipOffsetY += 1.75 * mipSize;
        mipOffsetX += 1.25 * mipSize;
        mipSize /= 2;
      }
      offset2 += 2 * size;
      if (size > 16) size /= 2;
    }
  };

  PMREMCubeUVPacker.prototype = {
    constructor: PMREMCubeUVPacker,

    update: function(renderer) {
      const size = this.cubeLods[0].width * 4;
      // top and bottom are swapped for some reason?
      camera.left = -size * 0.5;
      camera.right = size * 0.5;
      camera.top = -size * 0.5;
      camera.bottom = size * 0.5;
      camera.near = 0;
      camera.far = 1;
      camera.updateProjectionMatrix();

      for (let i = 0; i < this.objects.length; i++) {
        scene.add(this.objects[i]);
      }

      const gammaInput = renderer.gammaInput;
      const gammaOutput = renderer.gammaOutput;
      const toneMapping = renderer.toneMapping;
      const toneMappingExposure = renderer.toneMappingExposure;
      const currentRenderTarget = renderer.getRenderTarget();

      renderer.gammaInput = false;
      renderer.gammaOutput = false;
      renderer.toneMapping = THREE.LinearToneMapping;
      renderer.toneMappingExposure = 1.0;
      renderer.setRenderTarget(this.CubeUVRenderTarget);
      renderer.render(scene, camera);

      renderer.setRenderTarget(currentRenderTarget);
      renderer.toneMapping = toneMapping;
      renderer.toneMappingExposure = toneMappingExposure;
      renderer.gammaInput = gammaInput;
      renderer.gammaOutput = gammaOutput;

      for (let i = 0; i < this.objects.length; i++) {
        scene.remove(this.objects[i]);
      }
    },

    dispose: function() {
      for (let i = 0, l = this.objects.length; i < l; i++) {
        this.objects[i].material.dispose();
      }

      this.objects[0].geometry.dispose();
    }
  };

  function getShader() {
    const shaderMaterial = new THREE.ShaderMaterial({
      uniforms: {
        faceIndex: { value: 0 },
        mapSize: { value: 0 },
        envMap: { value: null },
        testColor: { value: new THREE.Vector3(1, 1, 1) }
      },

      vertexShader:
        "precision highp float;\
        varying vec2 vUv;\
        void main() {\
          vUv = uv;\
          gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\
        }",

      fragmentShader:
        "precision highp float;\
        varying vec2 vUv;\
        uniform samplerCube envMap;\
        uniform float mapSize;\
        uniform vec3 testColor;\
        uniform int faceIndex;\
        \
        void main() {\
          vec3 sampleDirection;\
          vec2 uv = vUv;\
          uv = uv * 2.0 - 1.0;\
          uv.y *= -1.0;\
          if(faceIndex == 0) {\
            sampleDirection = normalize(vec3(1.0, uv.y, -uv.x));\
          } else if(faceIndex == 1) {\
            sampleDirection = normalize(vec3(uv.x, 1.0, uv.y));\
          } else if(faceIndex == 2) {\
            sampleDirection = normalize(vec3(uv.x, uv.y, 1.0));\
          } else if(faceIndex == 3) {\
            sampleDirection = normalize(vec3(-1.0, uv.y, uv.x));\
          } else if(faceIndex == 4) {\
            sampleDirection = normalize(vec3(uv.x, -1.0, -uv.y));\
          } else {\
            sampleDirection = normalize(vec3(-uv.x, uv.y, -1.0));\
          }\
          vec4 color = envMapTexelToLinear( textureCube( envMap, sampleDirection ) );\
          gl_FragColor = linearToOutputTexel( color );\
        }",

      blending: THREE.NoBlending
    });

    shaderMaterial.type = "PMREMCubeUVPacker";

    return shaderMaterial;
  }

  return PMREMCubeUVPacker;
})();

/**
 * @author Prashant Sharma / spidersharma03
 * @author Ben Houston / bhouston, https://clara.io
 *
 * To avoid cube map seams, I create an extra pixel around each face. This way when the cube map is
 * sampled by an application later(with a little care by sampling the centre of the texel), the extra 1 border
 *	of pixels makes sure that there is no seams artifacts present. This works perfectly for cubeUV format as
 *	well where the 6 faces can be arranged in any manner whatsoever.
 * Code in the beginning of fragment shader's main function does this job for a given resolution.
 *	Run Scene_PMREM_Test.html in the examples directory to see the sampling from the cube lods generated
 *	by this class.
 */

const PMREMGenerator = (function() {
  const shader = getShader();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.0, 1000);
  const scene = new THREE.Scene();
  const planeMesh = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2, 0), shader);
  planeMesh.material.side = THREE.DoubleSide;
  scene.add(planeMesh);
  scene.add(camera);

  const PMREMGenerator = function(sourceTexture, samplesPerLevel, resolution) {
    this.sourceTexture = sourceTexture;
    this.resolution = resolution !== undefined ? resolution : 256; // NODE: 256 is currently hard coded in the glsl code for performance reasons
    this.samplesPerLevel = samplesPerLevel !== undefined ? samplesPerLevel : 32;

    const monotonicEncoding =
      this.sourceTexture.encoding === THREE.LinearEncoding ||
      this.sourceTexture.encoding === THREE.GammaEncoding ||
      this.sourceTexture.encoding === THREE.sRGBEncoding;

    this.sourceTexture.minFilter = monotonicEncoding ? THREE.LinearFilter : THREE.NearestFilter;
    this.sourceTexture.magFilter = monotonicEncoding ? THREE.LinearFilter : THREE.NearestFilter;
    this.sourceTexture.generateMipmaps = this.sourceTexture.generateMipmaps && monotonicEncoding;

    this.cubeLods = [];

    let size = this.resolution;
    const params = {
      format: this.sourceTexture.format,
      magFilter: this.sourceTexture.magFilter,
      minFilter: this.sourceTexture.minFilter,
      type: this.sourceTexture.type,
      generateMipmaps: this.sourceTexture.generateMipmaps,
      anisotropy: this.sourceTexture.anisotropy,
      encoding: this.sourceTexture.encoding
    };

    // how many LODs fit in the given CubeUV Texture.
    this.numLods = Math.log(size) / Math.log(2) - 2; // IE11 doesn't support Math.log2

    for (let i = 0; i < this.numLods; i++) {
      const renderTarget = new THREE.WebGLRenderTargetCube(size, size, params);
      renderTarget.texture.name = "PMREMGenerator.cube" + i;
      this.cubeLods.push(renderTarget);
      size = Math.max(16, size / 2);
    }
  };

  PMREMGenerator.prototype = {
    constructor: PMREMGenerator,

    /*
     * Prashant Sharma / spidersharma03: More thought and work is needed here.
     * Right now it's a kind of a hack to use the previously convolved map to convolve the current one.
     * I tried to use the original map to convolve all the lods, but for many textures(specially the high frequency)
     * even a high number of samples(1024) dosen't lead to satisfactory results.
     * By using the previous convolved maps, a lower number of samples are generally sufficient(right now 32, which
     * gives okay results unless we see the reflection very carefully, or zoom in too much), however the math
     * goes wrong as the distribution function tries to sample a larger area than what it should be. So I simply scaled
     * the roughness by 0.9(totally empirical) to try to visually match the original result.
     * The condition "if(i <5)" is also an attemt to make the result match the original result.
     * This method requires the most amount of thinking I guess. Here is a paper which we could try to implement in future::
     * https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
     */
    update: function(renderer) {
      // Texture should only be flipped for CubeTexture, not for
      // a Texture created via THREE.WebGLRenderTargetCube.
      const tFlip = this.sourceTexture.isCubeTexture ? -1 : 1;

      shader.defines["SAMPLES_PER_LEVEL"] = this.samplesPerLevel;
      shader.uniforms["faceIndex"].value = 0;
      shader.uniforms["envMap"].value = this.sourceTexture;
      shader.envMap = this.sourceTexture;
      shader.needsUpdate = true;

      const gammaInput = renderer.gammaInput;
      const gammaOutput = renderer.gammaOutput;
      const toneMapping = renderer.toneMapping;
      const toneMappingExposure = renderer.toneMappingExposure;
      const currentRenderTarget = renderer.getRenderTarget();

      renderer.toneMapping = THREE.LinearToneMapping;
      renderer.toneMappingExposure = 1.0;
      renderer.gammaInput = false;
      renderer.gammaOutput = false;

      for (let i = 0; i < this.numLods; i++) {
        const r = i / (this.numLods - 1);
        shader.uniforms["roughness"].value = r * 0.9; // see comment above, pragmatic choice
        // Only apply the tFlip for the first LOD
        shader.uniforms["tFlip"].value = i == 0 ? tFlip : 1;
        const size = this.cubeLods[i].width;
        shader.uniforms["mapSize"].value = size;
        this.renderToCubeMapTarget(renderer, this.cubeLods[i]);

        if (i < 5) shader.uniforms["envMap"].value = this.cubeLods[i].texture;
      }

      renderer.setRenderTarget(currentRenderTarget);
      renderer.toneMapping = toneMapping;
      renderer.toneMappingExposure = toneMappingExposure;
      renderer.gammaInput = gammaInput;
      renderer.gammaOutput = gammaOutput;
    },

    renderToCubeMapTarget: function(renderer, renderTarget) {
      for (let i = 0; i < 6; i++) {
        this.renderToCubeMapTargetFace(renderer, renderTarget, i);
      }
    },

    renderToCubeMapTargetFace: function(renderer, renderTarget, faceIndex) {
      shader.uniforms["faceIndex"].value = faceIndex;
      renderer.setRenderTarget(renderTarget, faceIndex);
      renderer.clear();
      renderer.render(scene, camera);
    },

    dispose: function() {
      for (let i = 0, l = this.cubeLods.length; i < l; i++) {
        this.cubeLods[i].dispose();
      }

      shader.dispose();
    }
  };

  function getShader() {
    const shaderMaterial = new ShaderMaterial({
      defines: {
        SAMPLES_PER_LEVEL: 20
      },

      uniforms: {
        faceIndex: { value: 0 },
        roughness: { value: 0.5 },
        mapSize: { value: 0.5 },
        envMap: { value: null },
        tFlip: { value: -1 }
      },

      vertexShader:
        "varying vec2 vUv;\n\
				void main() {\n\
					vUv = uv;\n\
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\
				}",

      fragmentShader:
        "#include <common>\n\
				varying vec2 vUv;\n\
				uniform int faceIndex;\n\
				uniform float roughness;\n\
				uniform samplerCube envMap;\n\
				uniform float mapSize;\n\
				uniform float tFlip;\n\
				\n\
				float GGXRoughnessToBlinnExponent( const in float ggxRoughness ) {\n\
					float a = ggxRoughness + 0.0001;\n\
					a *= a;\n\
					return ( 2.0 / a - 2.0 );\n\
				}\n\
				vec3 ImportanceSamplePhong(vec2 uv, mat3 vecSpace, float specPow) {\n\
					float phi = uv.y * 2.0 * PI;\n\
					float cosTheta = pow(1.0 - uv.x, 1.0 / (specPow + 1.0));\n\
					float sinTheta = sqrt(1.0 - cosTheta * cosTheta);\n\
					vec3 sampleDir = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);\n\
					return vecSpace * sampleDir;\n\
				}\n\
				vec3 ImportanceSampleGGX( vec2 uv, mat3 vecSpace, float Roughness )\n\
				{\n\
					float a = Roughness * Roughness;\n\
					float Phi = 2.0 * PI * uv.x;\n\
					float CosTheta = sqrt( (1.0 - uv.y) / ( 1.0 + (a*a - 1.0) * uv.y ) );\n\
					float SinTheta = sqrt( 1.0 - CosTheta * CosTheta );\n\
					return vecSpace * vec3(SinTheta * cos( Phi ), SinTheta * sin( Phi ), CosTheta);\n\
				}\n\
				mat3 matrixFromVector(vec3 n) {\n\
					float a = 1.0 / (1.0 + n.z);\n\
					float b = -n.x * n.y * a;\n\
					vec3 b1 = vec3(1.0 - n.x * n.x * a, b, -n.x);\n\
					vec3 b2 = vec3(b, 1.0 - n.y * n.y * a, -n.y);\n\
					return mat3(b1, b2, n);\n\
				}\n\
				\n\
				vec4 testColorMap(float Roughness) {\n\
					vec4 color;\n\
					if(faceIndex == 0)\n\
						color = vec4(1.0,0.0,0.0,1.0);\n\
					else if(faceIndex == 1)\n\
						color = vec4(0.0,1.0,0.0,1.0);\n\
					else if(faceIndex == 2)\n\
						color = vec4(0.0,0.0,1.0,1.0);\n\
					else if(faceIndex == 3)\n\
						color = vec4(1.0,1.0,0.0,1.0);\n\
					else if(faceIndex == 4)\n\
						color = vec4(0.0,1.0,1.0,1.0);\n\
					else\n\
						color = vec4(1.0,0.0,1.0,1.0);\n\
					color *= ( 1.0 - Roughness );\n\
					return color;\n\
				}\n\
				void main() {\n\
					vec3 sampleDirection;\n\
					vec2 uv = vUv*2.0 - 1.0;\n\
					float offset = -1.0/mapSize;\n\
					const float a = -1.0;\n\
					const float b = 1.0;\n\
					float c = -1.0 + offset;\n\
					float d = 1.0 - offset;\n\
					float bminusa = b - a;\n\
					uv.x = (uv.x - a)/bminusa * d - (uv.x - b)/bminusa * c;\n\
					uv.y = (uv.y - a)/bminusa * d - (uv.y - b)/bminusa * c;\n\
					if (faceIndex==0) {\n\
						sampleDirection = vec3(1.0, -uv.y, -uv.x);\n\
					} else if (faceIndex==1) {\n\
						sampleDirection = vec3(-1.0, -uv.y, uv.x);\n\
					} else if (faceIndex==2) {\n\
						sampleDirection = vec3(uv.x, 1.0, uv.y);\n\
					} else if (faceIndex==3) {\n\
						sampleDirection = vec3(uv.x, -1.0, -uv.y);\n\
					} else if (faceIndex==4) {\n\
						sampleDirection = vec3(uv.x, -uv.y, 1.0);\n\
					} else {\n\
						sampleDirection = vec3(-uv.x, -uv.y, -1.0);\n\
					}\n\
					vec3 correctedDirection = vec3( tFlip * sampleDirection.x, sampleDirection.yz );\n\
					mat3 vecSpace = matrixFromVector( normalize( correctedDirection ) );\n\
					vec3 rgbColor = vec3(0.0);\n\
					const int NumSamples = SAMPLES_PER_LEVEL;\n\
					vec3 vect;\n\
					float weight = 0.0;\n\
					for( int i = 0; i < NumSamples; i ++ ) {\n\
						float sini = sin(float(i));\n\
						float cosi = cos(float(i));\n\
						float r = rand(vec2(sini, cosi));\n\
						vect = ImportanceSampleGGX(vec2(float(i) / float(NumSamples), r), vecSpace, roughness);\n\
						float dotProd = dot(vect, normalize(sampleDirection));\n\
						weight += dotProd;\n\
						vec3 color = envMapTexelToLinear(textureCube(envMap, vect)).rgb;\n\
						rgbColor.rgb += color;\n\
					}\n\
					rgbColor /= float(NumSamples);\n\
					//rgbColor = testColorMap( roughness ).rgb;\n\
					gl_FragColor = linearToOutputTexel( vec4( rgbColor, 1.0 ) );\n\
				}",

      blending: NoBlending
    });

    shaderMaterial.type = "PMREMGenerator";

    return shaderMaterial;
  }

  return PMREMGenerator;
})();

/**
 * @author zz85 / https://github.com/zz85
 *
 * Based on "A Practical Analytic Model for Daylight"
 * aka The Preetham Model, the de facto standard analytic skydome model
 * http://www.cs.utah.edu/~shirley/papers/sunsky/sunsky.pdf
 *
 * First implemented by Simon Wallner
 * http://www.simonwallner.at/projects/atmospheric-scattering
 *
 * Improved by Martin Upitis
 * http://blenderartists.org/forum/showthread.php?245954-preethams-sky-impementation-HDR
 *
 * Three.js integration by zz85 http://twitter.com/blurspline
 */

const vertexShader = `
#include <common>
#include <fog_pars_vertex>

uniform vec3 sunPosition;
uniform float rayleigh;
uniform float turbidity;
uniform float mieCoefficient;

varying vec3 vWorldPosition;
varying vec3 vSunDirection;
varying float vSunfade;
varying vec3 vBetaR;
varying vec3 vBetaM;
varying float vSunE;

const vec3 up = vec3( 0.0, 1.0, 0.0 );

// constants for atmospheric scattering
const float e = 2.71828182845904523536028747135266249775724709369995957;
const float pi = 3.141592653589793238462643383279502884197169;

// wavelength of used primaries, according to preetham
const vec3 lambda = vec3( 680E-9, 550E-9, 450E-9 );
// this pre-calcuation replaces older TotalRayleigh(vec3 lambda) function:
// (8.0 * pow(pi, 3.0) * pow(pow(n, 2.0) - 1.0, 2.0) * (6.0 + 3.0 * pn)) / (3.0 * N * pow(lambda, vec3(4.0)) * (6.0 - 7.0 * pn))
const vec3 totalRayleigh = vec3( 5.804542996261093E-6, 1.3562911419845635E-5, 3.0265902468824876E-5 );

// mie stuff
// K coefficient for the primaries
const float v = 4.0;
const vec3 K = vec3( 0.686, 0.678, 0.666 );
// MieConst = pi * pow( ( 2.0 * pi ) / lambda, vec3( v - 2.0 ) ) * K
const vec3 MieConst = vec3( 1.8399918514433978E14, 2.7798023919660528E14, 4.0790479543861094E14 );

// earth shadow hack
// cutoffAngle = pi / 1.95;
const float cutoffAngle = 1.6110731556870734;
const float steepness = 1.5;
const float EE = 1000.0;

float sunIntensity( float zenithAngleCos ) {
  zenithAngleCos = clamp( zenithAngleCos, -1.0, 1.0 );
  return EE * max( 0.0, 1.0 - pow( e, -( ( cutoffAngle - acos( zenithAngleCos ) ) / steepness ) ) );
}

vec3 totalMie( float T ) {
  float c = ( 0.2 * T ) * 10E-18;
  return 0.434 * c * MieConst;
}

void main() {
  #include <begin_vertex>

  vec4 worldPosition = modelMatrix * vec4( position, 1.0 );
  vWorldPosition = worldPosition.xyz;

  #include <project_vertex>

  vSunDirection = normalize( sunPosition );

  vSunE = sunIntensity( dot( vSunDirection, up ) );

  vSunfade = 1.0 - clamp( 1.0 - exp( ( sunPosition.y / 450000.0 ) ), 0.0, 1.0 );

  float rayleighCoefficient = rayleigh - ( 1.0 * ( 1.0 - vSunfade ) );

  // extinction (absorbtion + out scattering)
  // rayleigh coefficients
  vBetaR = totalRayleigh * rayleighCoefficient;

  // mie coefficients
  vBetaM = totalMie( turbidity ) * mieCoefficient;

  #include <fog_vertex>
}
`;

const fragmentShader = `
#include <common>
#include <fog_pars_fragment>

varying vec3 vWorldPosition;
varying vec3 vSunDirection;
varying float vSunfade;
varying vec3 vBetaR;
varying vec3 vBetaM;
varying float vSunE;

uniform float luminance;
uniform float mieDirectionalG;

const vec3 cameraPos = vec3( 0.0, 0.0, 0.0 );

// constants for atmospheric scattering
const float pi = 3.141592653589793238462643383279502884197169;

const float n = 1.0003; // refractive index of air
const float N = 2.545E25; // number of molecules per unit volume for air at
// 288.15K and 1013mb (sea level -45 celsius)

// optical length at zenith for molecules
const float rayleighZenithLength = 8.4E3;
const float mieZenithLength = 1.25E3;
const vec3 up = vec3( 0.0, 1.0, 0.0 );
// 66 arc seconds -> degrees, and the cosine of that
const float sunAngularDiameterCos = 0.999956676946448443553574619906976478926848692873900859324;

// 3.0 / ( 16.0 * pi )
const float THREE_OVER_SIXTEENPI = 0.05968310365946075;
// 1.0 / ( 4.0 * pi )
const float ONE_OVER_FOURPI = 0.07957747154594767;

float rayleighPhase( float cosTheta ) {
  return THREE_OVER_SIXTEENPI * ( 1.0 + pow( cosTheta, 2.0 ) );
}

float hgPhase( float cosTheta, float g ) {
  float g2 = pow( g, 2.0 );
  float inverse = 1.0 / pow( 1.0 - 2.0 * g * cosTheta + g2, 1.5 );
  return ONE_OVER_FOURPI * ( ( 1.0 - g2 ) * inverse );
}

// Filmic ToneMapping http://filmicgames.com/archives/75
const float A = 0.15;
const float B = 0.50;
const float C = 0.10;
const float D = 0.20;
const float E = 0.02;
const float F = 0.30;

const float whiteScale = 1.0748724675633854; // 1.0 / Uncharted2Tonemap(1000.0)

vec3 Uncharted2Tonemap( vec3 x ) {
  return ( ( x * ( A * x + C * B ) + D * E ) / ( x * ( A * x + B ) + D * F ) ) - E / F;
}

void main() {
  // optical length
  // cutoff angle at 90 to avoid singularity in next formula.
  float zenithAngle = acos( max( 0.0, dot( up, normalize( vWorldPosition - cameraPos ) ) ) );
  float inverse = 1.0 / ( cos( zenithAngle ) + 0.15 * pow( 93.885 - ( ( zenithAngle * 180.0 ) / pi ), -1.253 ) );
  float sR = rayleighZenithLength * inverse;
  float sM = mieZenithLength * inverse;

  // combined extinction factor
  vec3 Fex = exp( -( vBetaR * sR + vBetaM * sM ) );

  // in scattering
  float cosTheta = dot( normalize( vWorldPosition - cameraPos ), vSunDirection );

  float rPhase = rayleighPhase( cosTheta * 0.5 + 0.5 );
  vec3 betaRTheta = vBetaR * rPhase;

  float mPhase = hgPhase( cosTheta, mieDirectionalG );
  vec3 betaMTheta = vBetaM * mPhase;

  vec3 Lin = pow( vSunE * ( ( betaRTheta + betaMTheta ) / ( vBetaR + vBetaM ) ) * ( 1.0 - Fex ), vec3( 1.5 ) );
  Lin *= mix( vec3( 1.0 ), pow( vSunE * ( ( betaRTheta + betaMTheta ) / ( vBetaR + vBetaM ) ) * Fex, vec3( 1.0 / 2.0 ) ), clamp( pow( 1.0 - dot( up, vSunDirection ), 5.0 ), 0.0, 1.0 ) );

  // nightsky
  vec3 direction = normalize( vWorldPosition - cameraPos );
  float theta = acos( direction.y ); // elevation --> y-axis, [-pi/2, pi/2]
  float phi = atan( direction.z, direction.x ); // azimuth --> x-axis [-pi/2, pi/2]
  vec2 uv = vec2( phi, theta ) / vec2( 2.0 * pi, pi ) + vec2( 0.5, 0.0 );
  vec3 L0 = vec3( 0.1 ) * Fex;

  // composition + solar disc
  float sundisk = smoothstep( sunAngularDiameterCos, sunAngularDiameterCos + 0.00002, cosTheta );
  L0 += ( vSunE * 19000.0 * Fex ) * sundisk;

  vec3 texColor = ( Lin + L0 ) * 0.04 + vec3( 0.0, 0.0003, 0.00075 );

  vec3 curr = Uncharted2Tonemap( ( log2( 2.0 / pow( luminance, 4.0 ) ) ) * texColor );
  vec3 color = curr * whiteScale;

  vec3 retColor = pow( color, vec3( 1.0 / ( 1.2 + ( 1.2 * vSunfade ) ) ) );

  gl_FragColor = vec4( retColor, 1.0 );

  #include <fog_fragment>
}
`;

export default class Sky extends Object3D {
  static shader = {
    uniforms: UniformsUtils.merge([
      UniformsLib.fog,
      {
        luminance: { value: 1 },
        turbidity: { value: 10 },
        rayleigh: { value: 2 },
        mieCoefficient: { value: 0.005 },
        mieDirectionalG: { value: 0.8 },
        sunPosition: { value: new Vector3() }
      }
    ]),
    vertexShader,
    fragmentShader
  };

  static _geometry = new BoxBufferGeometry(1, 1, 1);

  constructor() {
    super();

    const material = new ShaderMaterial({
      fragmentShader: Sky.shader.fragmentShader,
      vertexShader: Sky.shader.vertexShader,
      uniforms: UniformsUtils.clone(Sky.shader.uniforms),
      side: BackSide,
      fog: true
    });

    this.skyScene = new Scene();
    this.cubeCamera = new CubeCamera(1, 100000, 512);
    this.skyScene.add(this.cubeCamera);

    this.sky = new Mesh(Sky._geometry, material);
    this.sky.name = "Sky";
    this.add(this.sky);

    this._inclination = 0;
    this._azimuth = 0.15;
    this._distance = 8000;
    this.updateSunPosition();
  }

  get turbidity() {
    return this.sky.material.uniforms.turbidity.value;
  }

  set turbidity(value) {
    this.sky.material.uniforms.turbidity.value = value;
  }

  get rayleigh() {
    return this.sky.material.uniforms.rayleigh.value;
  }

  set rayleigh(value) {
    this.sky.material.uniforms.rayleigh.value = value;
  }

  get luminance() {
    return this.sky.material.uniforms.luminance.value;
  }

  set luminance(value) {
    this.sky.material.uniforms.luminance.value = value;
  }

  get mieCoefficient() {
    return this.sky.material.uniforms.mieCoefficient.value;
  }

  set mieCoefficient(value) {
    this.sky.material.uniforms.mieCoefficient.value = value;
  }

  get mieDirectionalG() {
    return this.sky.material.uniforms.mieDirectionalG.value;
  }

  set mieDirectionalG(value) {
    this.sky.material.uniforms.mieDirectionalG.value = value;
  }

  get inclination() {
    return this._inclination;
  }

  set inclination(value) {
    this._inclination = value;
    this.updateSunPosition();
  }

  get azimuth() {
    return this._azimuth;
  }

  set azimuth(value) {
    this._azimuth = value;
    this.updateSunPosition();
  }

  get distance() {
    return this._distance;
  }

  set distance(value) {
    this._distance = value;
    this.updateSunPosition();
  }

  updateSunPosition() {
    const theta = Math.PI * (this._inclination - 0.5);
    const phi = 2 * Math.PI * (this._azimuth - 0.5);

    const distance = this._distance;

    const x = distance * Math.cos(phi);
    const y = distance * Math.sin(phi) * Math.sin(theta);
    const z = distance * Math.sin(phi) * Math.cos(theta);

    this.sky.material.uniforms.sunPosition.value.set(x, y, z).normalize();
    this.sky.scale.set(distance, distance, distance);
  }

  generateEnvironmentMap(renderer) {
    this.skyScene.add(this.sky);
    this.cubeCamera.update(renderer, this.skyScene);
    this.add(this.sky);
    const vrEnabled = renderer.vr.enabled;
    renderer.vr.enabled = false;
    const pmremGenerator = new PMREMGenerator(this.cubeCamera.renderTarget.texture);
    pmremGenerator.update(renderer);
    const pmremCubeUVPacker = new PMREMCubeUVPacker(pmremGenerator.cubeLods);
    pmremCubeUVPacker.update(renderer);
    renderer.vr.enabled = vrEnabled;
    pmremGenerator.dispose();
    pmremCubeUVPacker.dispose();
    return pmremCubeUVPacker.CubeUVRenderTarget.texture;
  }

  copy(source, recursive = true) {
    if (recursive) {
      this.remove(this.sky);
    }

    super.copy(source, recursive);

    if (recursive) {
      const skyIndex = source.children.indexOf(source.sky);

      if (skyIndex !== -1) {
        this.sky = this.children[skyIndex];
      }
    }

    this.turbidity = source.turbidity;
    this.rayleigh = source.rayleigh;
    this.luminance = source.luminance;
    this.mieCoefficient = source.mieCoefficient;
    this.mieDirectionalG = source.mieDirectionalG;
    this.inclination = source.inclination;
    this.azimuth = source.azimuth;
    this.distance = source.distance;

    return this;
  }
}
