<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="ng[.~'^(No|no|NO) ']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="neg">yes</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="ng[.~'^(No|no|NO) ' or @neg='yes']//ng">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="neg">yes</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<!-- "No subdural or extradural." -->
<xsl:template match="ag[not(ancestor::ng) and preceding-sibling::*[.~'^(No|no|NO|\?)$']]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="neg">yes</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<!-- "? haemorrhagic stroke." -->
<xsl:template match="ng[not(ancestor::ng) and preceding-sibling::*[1][.='?']]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="neg">yes</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<!-- "? haemorrhagic stroke." where the ? lost between two s elements -->
<xsl:template match="ng[not(ancestor::ng) and ancestor::s[preceding-sibling::*[1][self::w[.='?']]]]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="neg">yes</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<!-- "Bleed?" -->
<xsl:template match="ng[not(ancestor::ng) and not(preceding-sibling) and following-sibling::*[1][.='?']]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="neg">yes</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>

