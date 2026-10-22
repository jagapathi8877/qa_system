/* ------------------------------ opening intro ---------------------------- */

(() => {
  const intro = document.getElementById("introScreen");
  const appShell = document.getElementById("appShell");
  const status = document.getElementById("introStatus");
  const glass = document.getElementById("introGlass");
  const shardLayer = document.getElementById("glassShards");
  const impact = document.getElementById("glassImpact");
  const crack = document.getElementById("glassCrack");
  const introRobots = [
    document.getElementById("introRobotLeft"),
    document.getElementById("introRobotRight"),
  ].filter(Boolean);
  const logoTracePaths = [
    document.querySelector("#introLogoTraceLeft path"),
    document.querySelector("#introLogoTraceRight path"),
  ].filter(Boolean);
  const headerRobot = document.getElementById("chatbotAnimation");
  const root = document.documentElement;

  if (!intro || !appShell || !status || !glass || !shardLayer || !impact || !crack) {
    appShell?.removeAttribute("inert");
    appShell?.removeAttribute("aria-hidden");
    root.classList.remove("intro-active");
    return;
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const timers = [];
  let finished = false;
  let shattering = false;
  let robotOrbitFrame = 0;

  function animateRobotToHeader() {
    if (!introRobots.length || !headerRobot || typeof introRobots[0].animate !== "function") return;
    if (robotOrbitFrame) cancelAnimationFrame(robotOrbitFrame);
    robotOrbitFrame = 0;
    const target = headerRobot.getBoundingClientRect();
    if (!target.width || !target.height) return;
    const targetX = target.left + target.width / 2;
    const targetY = target.top + target.height / 2;
    const robotRects = introRobots.map((robot) => robot.getBoundingClientRect());
    const movingIndex = robotRects.reduce((closestIndex, rect, index) => {
      const closest = robotRects[closestIndex];
      return Math.hypot(rect.left + rect.width / 2 - targetX, rect.top + rect.height / 2 - targetY)
        < Math.hypot(closest.left + closest.width / 2 - targetX, closest.top + closest.height / 2 - targetY)
        ? index
        : closestIndex;
    }, 0);
    const movingRobot = introRobots[movingIndex];
    const source = robotRects[movingIndex];
    if (!source.width || !source.height) return;
    headerRobot.style.animationPlayState = "paused";
    introRobots.forEach((robot, index) => {
      if (index === movingIndex) return;
      robot.animate(
        [{ opacity: 1 }, { opacity: 0 }],
        { duration: 360, easing: "ease-out", fill: "forwards" }
      );
    });
    movingRobot.animate(
      [
        {
          left: `${source.left + source.width / 2}px`,
          top: `${source.top + source.height / 2}px`,
          width: `${source.width}px`,
          height: `${source.height}px`,
        },
        {
          left: `${target.left + target.width / 2}px`,
          top: `${target.top + target.height / 2}px`,
          width: `${target.width}px`,
          height: `${target.height}px`,
        },
      ],
      { duration: 650, easing: "cubic-bezier(0.22, 0.8, 0.25, 1)", fill: "forwards" }
    );
  }

  function traceLogo(startTime) {
    if (introRobots.length !== 2 || logoTracePaths.length !== 2) return;
    const cycleMs = 3200;
    introRobots.forEach((robot, index) => {
      const path = logoTracePaths[index];
      const pathLength = path.getTotalLength();
      const phase = (startTime % cycleMs) / cycleMs;
      const progress = (1 - Math.cos(phase * Math.PI * 2)) / 2;
      const point = path.getPointAtLength(pathLength * progress);
      const matrix = path.getScreenCTM();
      if (!matrix) return;
      const svgPoint = path.ownerSVGElement.createSVGPoint();
      svgPoint.x = point.x;
      svgPoint.y = point.y;
      const screenPoint = svgPoint.matrixTransform(matrix);
      robot.style.left = `${screenPoint.x}px`;
      robot.style.top = `${screenPoint.y}px`;
    });
    robotOrbitFrame = requestAnimationFrame(traceLogo);
  }

  function buildGlassShards() {
    const namespace = "http://www.w3.org/2000/svg";
    const columns = 5;
    const rows = 4;
    const points = [];

    for (let row = 0; row <= rows; row += 1) {
      points[row] = [];
      for (let column = 0; column <= columns; column += 1) {
        const edge = row === 0 || row === rows || column === 0 || column === columns;
        points[row][column] = {
          x: (column / columns) * 1000 + (edge ? 0 : (Math.random() - 0.5) * 48),
          y: (row / rows) * 700 + (edge ? 0 : (Math.random() - 0.5) * 44),
        };
      }
    }

    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const topLeft = points[row][column];
        const topRight = points[row][column + 1];
        const bottomLeft = points[row + 1][column];
        const bottomRight = points[row + 1][column + 1];
        const triangles = Math.random() > 0.5
          ? [[topLeft, topRight, bottomRight], [topLeft, bottomRight, bottomLeft]]
          : [[topLeft, topRight, bottomLeft], [topRight, bottomRight, bottomLeft]];

        triangles.forEach((vertices) => {
          const polygon = document.createElementNS(namespace, "polygon");
          polygon.classList.add("glass-shard");
          polygon.setAttribute("points", vertices.map(({ x, y }) => `${x},${y}`).join(" "));
          polygon.dataset.centerX = String(vertices.reduce((sum, point) => sum + point.x, 0) / 3);
          polygon.dataset.centerY = String(vertices.reduce((sum, point) => sum + point.y, 0) / 3);
          shardLayer.appendChild(polygon);
        });
      }
    }
  }

  function createCrackPath(x, y) {
    const paths = [];
    for (let ray = 0; ray < 15; ray += 1) {
      const angle = (Math.PI * 2 * ray) / 15 + (Math.random() - 0.5) * 0.18;
      const length = 70 + Math.random() * 210;
      const middleX = x + Math.cos(angle + 0.12) * length * 0.52;
      const middleY = y + Math.sin(angle + 0.12) * length * 0.52;
      const endX = x + Math.cos(angle) * length;
      const endY = y + Math.sin(angle) * length;
      paths.push(`M ${x} ${y} L ${middleX} ${middleY} L ${endX} ${endY}`);
    }
    return paths.join(" ");
  }

  function burstGlass(event) {
    if (finished || shattering) return;
    shattering = true;

    if (reducedMotion) {
      finishIntro();
      return;
    }

    animateRobotToHeader();

    const rect = glass.getBoundingClientRect();
    const hasPointerPosition = event && Number.isFinite(event.clientX) && Number.isFinite(event.clientY);
    const impactX = hasPointerPosition
      ? Math.max(0, Math.min(1000, ((event.clientX - rect.left) / rect.width) * 1000))
      : 500;
    const impactY = hasPointerPosition
      ? Math.max(0, Math.min(700, ((event.clientY - rect.top) / rect.height) * 700))
      : 350;
    const polygons = shardLayer.querySelectorAll(".glass-shard");

    timers.forEach(clearTimeout);
    intro.classList.add("is-shattering");
    impact.setAttribute("cx", String(impactX));
    impact.setAttribute("cy", String(impactY));
    impact.setAttribute("r", "270");
    impact.style.setProperty("--impact-x", `${impactX}px`);
    impact.style.setProperty("--impact-y", `${impactY}px`);
    crack.setAttribute("d", createCrackPath(impactX, impactY));
    impact.classList.add("is-flashing");
    crack.classList.add("is-spreading");

    polygons.forEach((polygon) => {
      const dx = Number(polygon.dataset.centerX) - impactX;
      const dy = Number(polygon.dataset.centerY) - impactY;
      const distance = Math.hypot(dx, dy) || 1;
      const travel = 300 + Math.random() * 490;
      polygon.style.setProperty("--shard-x", `${(dx / distance) * travel + (Math.random() - 0.5) * 120}px`);
      polygon.style.setProperty("--shard-y", `${(dy / distance) * travel + (Math.random() - 0.5) * 120}px`);
      polygon.style.setProperty("--shard-rotation", `${(Math.random() - 0.5) * 520}deg`);
      polygon.style.setProperty("--shard-delay", `${Math.random() * 70}ms`);
      polygon.classList.add("is-flying");
    });

    timers.push(setTimeout(finishIntro, 680));
  }

  function updateStatus(message) {
    status.classList.add("is-changing");
    timers.push(setTimeout(() => {
      status.textContent = message;
      status.classList.remove("is-changing");
    }, 180));
  }

  function finishIntro() {
    if (finished) return;
    finished = true;
    if (robotOrbitFrame) cancelAnimationFrame(robotOrbitFrame);
    robotOrbitFrame = 0;
    timers.forEach(clearTimeout);
    root.classList.remove("intro-active");
    root.classList.add("intro-leaving");
    appShell.removeAttribute("inert");
    appShell.removeAttribute("aria-hidden");

    timers.push(setTimeout(() => {
      intro.remove();
      if (headerRobot) headerRobot.style.animationPlayState = "";
      root.classList.remove("intro-leaving");
      document.getElementById("questionInput")?.focus({ preventScroll: true });
    }, reducedMotion ? 40 : 760));
  }

  buildGlassShards();
  if (!reducedMotion) robotOrbitFrame = requestAnimationFrame(traceLogo);
  intro.addEventListener("click", burstGlass);
  document.addEventListener("keydown", (event) => {
    if (finished || !intro.isConnected) return;
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    burstGlass();
  });
  intro.focus({ preventScroll: true });

  const firstMessageAt = reducedMotion ? 260 : 820;
  const readyMessageAt = reducedMotion ? 520 : 1640;

  timers.push(setTimeout(() => updateStatus("Finding the right answers..."), firstMessageAt));
  timers.push(setTimeout(() => updateStatus("Ready."), readyMessageAt));
})();
