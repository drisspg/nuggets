import fs from "node:fs/promises"
import { Resvg } from "@resvg/resvg-js"
import satori from "satori"
import { BuildCtx } from "../../util/ctx"
import { GlobalConfiguration } from "../../cfg"
import { QuartzPluginData } from "../vfile"
import { SOCIAL_HEIGHT, SOCIAL_WIDTH, SocialMetadata, socialMetadata } from "../../util/social"
import { write } from "./helpers"

const paper = "#f4f0e8"
const ink = "#1a2e22"
const green = "#5f7f67"
const amber = "#b98049"
const rule = "#d0ccc4"

const dataImage = (svg: string) =>
  `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`

function illustration(kind: SocialMetadata["artwork"]) {
  const lines: string[] = []
  for (let n = 20; n < 380; n += 24) {
    lines.push(`<path d="M${n} 0V360 M0 ${n}H380" stroke="${rule}" opacity=".4"/>`)
  }
  if (kind === "attention") {
    for (let n = 0; n < 8; n++) {
      const x = 35 + n * 44
      const color = n < 4 ? green : n === 4 ? amber : "#aa413a"
      for (const y of [52, 308]) {
        const end = y < 180 ? y + 19 : y - 19
        lines.push(
          `<path d="M211 180Q${x} 180 ${x} ${end}" fill="none" stroke="${color}" stroke-width="1.6" opacity=".7"/>`,
        )
        lines.push(
          `<rect x="${x - 15}" y="${y - 19}" width="30" height="38" rx="3" fill="${paper}" stroke="${color}" stroke-width="1.5"/>`,
        )
        for (let bit = 0; bit < 3; bit++) {
          lines.push(
            `<path d="M${x - 8} ${y - 8 + bit * 8}h${bit === n % 3 ? 16 : 10}" stroke="${color}" stroke-width="3" opacity=".6"/>`,
          )
        }
      }
    }
    lines.push(
      `<circle cx="211" cy="180" r="42" fill="${paper}"/><circle cx="211" cy="180" r="28" fill="${amber}"/><circle cx="211" cy="180" r="35" fill="none" stroke="${amber}" opacity=".5"/><path d="M204 180h14m-7-7v14" stroke="${paper}" stroke-width="2.5"/>`,
    )
  } else if (kind === "tiles") {
    for (let row = 0; row < 7; row++) {
      for (let col = 0; col < 7; col++) {
        const active = col <= row
        const x = 47 + col * 42
        const y = 45 + row * 42
        lines.push(
          `<rect x="${x}" y="${y}" width="32" height="32" rx="3" fill="${row === col ? amber : active ? green : paper}" fill-opacity="${row === col ? 1 : active ? 0.22 + row * 0.055 : 1}" stroke="${active ? green : rule}" stroke-width="1"/>`,
        )
      }
    }
    lines.push(
      `<path d="M30 44v302h306" fill="none" stroke="${ink}" stroke-width="1.5"/><path d="m331 341 5 5-5 5" fill="none" stroke="${ink}" stroke-width="1.5"/>`,
    )
  } else {
    const nodes = [
      [86, 83],
      [273, 62],
      [190, 170],
      [61, 268],
      [292, 289],
    ]
    for (const [i, [x, y]] of nodes.entries()) {
      for (const [xx, yy] of nodes.slice(i + 1)) {
        lines.push(
          `<path d="M${x} ${y}Q190 180 ${xx} ${yy}" fill="none" stroke="${i % 2 ? amber : green}" stroke-width="1.5" opacity=".4"/>`,
        )
      }
      lines.push(
        `<circle cx="${x}" cy="${y}" r="${i === 2 ? 56 : 34}" fill="${paper}"/><circle cx="${x}" cy="${y}" r="${i === 2 ? 45 : 24}" fill="${i % 2 ? amber : green}"/><circle cx="${x}" cy="${y}" r="${i === 2 ? 56 : 34}" fill="none" stroke="${i % 2 ? amber : green}" opacity=".45"/><path d="M${x - 7} ${y}h14${i % 2 ? "" : `m-7-7v14`}" stroke="${paper}" stroke-width="2"/>`,
      )
    }
  }
  return dataImage(
    `<svg xmlns="http://www.w3.org/2000/svg" width="380" height="360" viewBox="0 0 380 360">${lines.join("")}</svg>`,
  )
}

// Font bytes are bundled with the source: image generation needs no network or system fonts.
const assets = Promise.all([
  fs.readFile("quartz/fonts/IBMPlexSans-Regular.ttf"),
  fs.readFile("quartz/fonts/IBMPlexSans-SemiBold.ttf"),
  fs.readFile("quartz/static/icon.svg", "utf8"),
])

export async function renderSocialImage(card: SocialMetadata) {
  const [regular, semibold, icon] = await assets
  const titleSize = card.title.length > 70 ? 50 : card.title.length > 45 ? 58 : 72
  const svg = await satori(
    (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          width: "100%",
          height: "100%",
          background: paper,
          color: ink,
          fontFamily: "Plex",
          padding: "48px 60px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <img src={dataImage(icon)} width={42} height={42} />
            <span style={{ fontSize: 27, fontWeight: 600, letterSpacing: 3 }}>
              {card.site.toUpperCase()}
            </span>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              fontSize: 16,
              letterSpacing: 2,
              color: green,
            }}
          >
            <div style={{ width: 7, height: 7, borderRadius: 7, background: amber }} />
            {card.section}
          </div>
        </div>
        {card.cover ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              flex: 1,
              gap: 22,
            }}
          >
            <div
              style={{
                fontSize: card.title.length > 60 ? 50 : 62,
                fontWeight: 600,
                lineHeight: 1.05,
                letterSpacing: -1.7,
                textWrap: "balance",
              }}
            >
              {card.title}
            </div>
            <img
              src={card.cover.dataUrl}
              width={1080}
              height={286}
              style={{ objectFit: "contain" }}
            />
          </div>
        ) : (
          <div style={{ display: "flex", alignItems: "center", flex: 1, gap: 26 }}>
            <div style={{ display: "flex", flexDirection: "column", width: 674 }}>
              <div
                style={{
                  fontSize: titleSize,
                  fontWeight: 600,
                  lineHeight: 1.06,
                  letterSpacing: -2.2,
                  textWrap: "balance",
                }}
              >
                {card.title}
              </div>
              <div
                style={{
                  display: "flex",
                  marginTop: 24,
                  maxWidth: 628,
                  fontSize: 24,
                  lineHeight: 1.35,
                  color: "#526558",
                }}
              >
                {card.description}
              </div>
            </div>
            <img src={illustration(card.artwork)} width={380} height={360} />
          </div>
        )}
      </div>
    ) as unknown as Parameters<typeof satori>[0],
    {
      width: SOCIAL_WIDTH,
      height: SOCIAL_HEIGHT,
      fonts: [
        { name: "Plex", data: regular, weight: 400, style: "normal" },
        { name: "Plex", data: semibold, weight: 600, style: "normal" },
      ],
    },
  )
  return new Resvg(svg, { font: { loadSystemFonts: false } }).render().asPng()
}

export async function writeSocialImage(
  ctx: BuildCtx,
  cfg: GlobalConfiguration,
  file: QuartzPluginData,
) {
  const card = socialMetadata(cfg, file)
  return write({ ctx, slug: card.imageSlug, ext: ".png", content: await renderSocialImage(card) })
}
