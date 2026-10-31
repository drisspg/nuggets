// Quoted titles and slash-delimited character highlights may themselves contain a flag's text.
const metaTokens = /(?:[^\s"'/]+|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\/(?:\\.|[^/\\])*\/|\/)+/g

/** Remove a bare word such as `annotate` from fence metadata, reporting whether it was present. */
export function takeFenceFlag(meta: string | null | undefined, flag: string) {
  let present = false
  const rest = meta?.replace(metaTokens, (token) => {
    if (token !== flag) return token
    present = true
    return ""
  })
  return { present, meta: rest }
}
