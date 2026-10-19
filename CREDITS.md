# Credits and licenses

drift-school is MIT licensed (see `LICENSE`). It is built on these open-source packages, used unmodified
from npm and bundled by Vite.

| Package | Version | License | Used for |
| --- | --- | --- | --- |
| [TensorFlow.js](https://github.com/tensorflow/tfjs) | 4.22.0 | Apache-2.0 | Policy and value networks, PPO updates (CPU backend) |
| [Rapier](https://github.com/dimforge/rapier.js) (`@dimforge/rapier3d-compat`) | 0.14.0 | Apache-2.0 | Wall collisions for the drivable car on the drive page |
| [three.js](https://github.com/mrdoob/three.js) | 0.170.0 | MIT | Rendering |
| [uPlot](https://github.com/leeoniya/uPlot) | 1.6.31 | MIT | Training charts |
| [fflate](https://github.com/101arrowz/fflate) | 0.8.2 | MIT | Compressing shared tracks in the URL hash |

Development tooling (not shipped): Vite, Vitest, vite-node, Playwright, ESLint, TypeScript.

## Techniques

- Proximal Policy Optimization with generalized advantage estimation: Schulman et al., 2017 and 2015.
- Pacejka "magic formula" tyre curve, in a simplified form: H. B. Pacejka, *Tyre and Vehicle Dynamics*.
- sfc32 random number generator: Chris Doty-Humphrey's PractRand test suite.
- Ramer-Douglas-Peucker simplification, Chaikin corner cutting and centripetal Catmull-Rom splines, used by the
  track editor.
- `public/coi-serviceworker.js` follows the well-known cross-origin-isolation service worker approach
  (for example [gzuidhof/coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker), MIT) so that GitHub
  Pages, which cannot set response headers, can still serve COOP and COEP.

## Assets

The car, track and environment models are generated in code. There are no bundled fonts, images or audio from
third parties. The screenshots and GIF in `docs/media` are captures of this app.
