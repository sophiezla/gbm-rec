import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { RoleStyle } from "./roles";

/** Lit, rotatable meshes on a dark background (plan Step 5.3). */
export class Viewer3D {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private loader = new GLTFLoader();
  private meshes = new Map<string, THREE.Object3D>();

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    host.appendChild(this.renderer.domElement);
    this.scene.background = new THREE.Color(0x111316);
    this.camera = new THREE.PerspectiveCamera(40, 1, 1, 5000);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(1, 1, 2);
    this.camera.add(key);
    this.scene.add(this.camera);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.listenToKeyEvents(host);   // arrow keys pan; see onKey for rotate/zoom
    host.addEventListener("keydown", (e) => this.onKey(e));
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  private resize(): void {
    const w = this.host.clientWidth || 1;
    const h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private onKey(e: KeyboardEvent): void {
    const step = Math.PI / 24;
    const offset = this.camera.position.clone().sub(this.controls.target);
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      offset.applyAxisAngle(this.camera.up, e.key === "ArrowLeft" ? step : -step);
    } else if (e.key === "+" || e.key === "=" || e.key === "-") {
      offset.multiplyScalar(e.key === "-" ? 1.1 : 0.9);
    } else {
      return;
    }
    e.preventDefault();
    this.camera.position.copy(this.controls.target).add(offset);
  }

  clear(): void {
    for (const obj of this.meshes.values()) this.scene.remove(obj);
    this.meshes.clear();
  }

  async add(key: string, url: string, style: RoleStyle, visible: boolean): Promise<void> {
    const gltf = await this.loader.loadAsync(url);
    const material = new THREE.MeshStandardMaterial({
      color: style.color,
      transparent: style.opacity < 1,
      opacity: style.opacity,
      depthWrite: style.opacity >= 0.5,
      side: THREE.DoubleSide,
    });
    gltf.scene.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) (o as THREE.Mesh).material = material;
    });
    gltf.scene.visible = visible;
    this.meshes.set(key, gltf.scene);
    this.scene.add(gltf.scene);
  }

  setVisible(key: string, visible: boolean): void {
    const obj = this.meshes.get(key);
    if (obj) obj.visible = visible;
  }

  has(key: string): boolean {
    return this.meshes.has(key);
  }

  /** Show exactly these layers and hide the rest. */
  showOnly(keys: Set<string>): void {
    for (const [key, obj] of this.meshes) obj.visible = keys.has(key);
  }

  /** Call ``fn`` whenever the user moves this camera (Compare links two views). */
  onCameraChange(fn: () => void): void {
    this.controls.addEventListener("change", fn);
  }

  /** Copy another viewer's camera exactly, so two views stay linked. */
  followCamera(other: Viewer3D): void {
    this.camera.position.copy(other.camera.position);
    this.camera.up.copy(other.camera.up);
    this.controls.target.copy(other.controls.target);
    this.camera.near = other.camera.near;
    this.camera.far = other.camera.far;
    this.camera.updateProjectionMatrix();
  }

  /** Frame the camera on one layer (the brain). */
  frame(key: string): void {
    const obj = this.meshes.get(key);
    if (!obj) return;
    const box = new THREE.Box3().setFromObject(obj);
    const centre = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    this.controls.target.copy(centre);
    // Patient frame is LPS; view from the front-left with superior up.
    this.camera.up.set(0, 0, 1);
    this.camera.position.copy(centre).add(new THREE.Vector3(-0.6, -1.2, 0.4).multiplyScalar(size));
    this.camera.near = size / 100;
    this.camera.far = size * 10;
    this.camera.updateProjectionMatrix();
  }
}
